#%%
# Import libraries
import pandas as pd 
import numpy as np 
import os
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce

#%%
# Loading Food Access Research Atlas dataset
food_access_url = "https://raw.githubusercontent.com/gguruksha/Datathon_GWU_2025/main/FoodAccessResearchAtlasData2019.csv"
food_access = pd.read_csv(food_access_url)

food_env_url ="https://github.com/gguruksha/Datathon_GWU_2025/raw/main/FoodEnvironmentAtlas.xls"
food_env = pd.read_excel(food_env_url)

#%%

# Quick peek at the datasets
print("📊 Food Access columns:", food_access.columns.tolist())
print("📋 Food Environment Atlas preview:\n", food_env.head())

#%%
#Checking available sheets in the Food Environment Excel file
xls = pd.ExcelFile('food_env')
print("📑 Available sheets:", xls.sheet_names)

#%%
#Loading relevant sheets
stores = pd.read_excel(xls, sheet_name='STORES')
health = pd.read_excel(xls, sheet_name='HEALTH')
socio = pd.read_excel(xls, sheet_name='SOCIOECONOMIC')
assistance = pd.read_excel(xls, sheet_name='ASSISTANCE')

#%%
#Dropping 'State' and 'County' columns from supplemental datasets to avoid duplication
for df in [health, socio, assistance]:
    df.drop(columns=['State', 'County'], inplace=True, errors='ignore')

#%%
# Previewing example sheet
    
print("✅ Stores columns:", stores.columns.tolist())
print("✅ Health sheet preview:\n", health.head())

#%%
#Checking  for missing values in food_access

print("🔍 Missing values in food_access:")
print(food_access.isnull().sum().sort_values(ascending=False).head(10))

#%%
# Merging environment-level data (all on FIPS)
dfs = [stores, health, socio, assistance]
food_env_combined = reduce(lambda left, right: pd.merge(left, right, on='FIPS', how='outer'), dfs)

print(f"✅ Combined food environment dataset shape: {food_env_combined.shape}")
print("Top missing columns in food_env_combined:")
print(food_env_combined.isnull().sum().sort_values(ascending=False).head(10))

#%%
# Merging county-level food environment data with tract-level food access data
food_access['CountyFIPS'] = food_access['CensusTract'].astype(str).str[:5].astype(int)
full_data = food_access.merge(food_env_combined, left_on='CountyFIPS', right_on='FIPS', how='left')

print("✅ Full merged dataset shape:", full_data.shape)
print(full_data.head())

#%%
#Dropping columns with >90% missing values
missing_ratio = full_data.isnull().mean()
cols_to_drop = missing_ratio[missing_ratio > 0.9].index
print(f"🧹 Dropping {len(cols_to_drop)} columns with >90% missing values...")
full_data.drop(columns=cols_to_drop, inplace=True)

#%%
# Visualize remaining missing values
plt.figure(figsize=(12, 6))
sns.heatmap(full_data.isnull(), cbar=False, yticklabels=False)
plt.title("🧯 Missing Values Heatmap (Post Merge & Cleanup)")
plt.tight_layout()
plt.show()


#%%
''' EDA'''
#%%
# EDA – Binary Target Distribution
full_data['LILATracts_1And10'].value_counts(dropna=False)

counts = full_data['LILATracts_1And10'].value_counts()
food_desert_percentage = (counts[1] / counts.sum()) * 100
print(f"Percentage of tracts that are food deserts: {food_desert_percentage:.2f}%")

#%%
# Univariate Summary of Key Features
univariate_features = ['POVRATE15', 'PCT_SNAP17', 'PCT_OBESE_ADULTS17', 'GROCPTH16']
print(full_data[univariate_features].describe().T)

#%%
# Bivariate Differences by Food Desert Status
bivariate = full_data.groupby('LILATracts_1And10')[univariate_features].mean().T
bivariate['Difference'] = bivariate[1.0] - bivariate[0.0]
print(bivariate)

#%%
# Grouped Mean Plot
import matplotlib.pyplot as plt
import seaborn as sns

grouped_means = full_data.groupby('LILATracts_1And10')[univariate_features].mean().T
grouped_means.columns = ['Not Food Desert', 'Food Desert']
grouped_means.plot(kind='bar', figsize=(10,6), rot=0)
plt.title('Average Socioeconomic & Access Factors by Food Desert Status')
plt.ylabel('Average Value')
plt.xlabel('Variable')
plt.legend(title='Census Tract Type')
plt.tight_layout()
plt.show()

#%%
# Boxplots to Visualize Distributions
titles = [
    'Poverty Rate (POVRATE15)',
    'SNAP Participation % (PCT_SNAP17)',
    'Obesity Rate % (PCT_OBESE_ADULTS17)',
    'Grocery Stores per 1000 Pop (GROCPTH16)'
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, feature in enumerate(univariate_features):
    sns.boxplot(data=full_data, x='LILATracts_1And10', y=feature, ax=axes.flatten()[i])
    axes.flatten()[i].set_title(titles[i])
    axes.flatten()[i].set_xlabel('Food Desert (0 = No, 1 = Yes)')
plt.suptitle('Boxplots of Key Indicators by Food Desert Status', fontsize=16)
plt.tight_layout()
plt.show()

#%%
# Correlation Heatmap with Top Drivers
numeric_data_clean = full_data.select_dtypes(include='number').dropna()
corr_series = numeric_data_clean.corr()['LILATracts_1And10'].sort_values(ascending=False)
print("Top Positive Correlations:\n", corr_series.head(10))
print("\nTop Negative Correlations:\n", corr_series.tail(10))

#%%
# Define model features based on correlation analysis
corr_features = [
    'LILATracts_1And10',
    'LowIncomeTracts',
    'lalowihalfshare',
    'lasnap10share',
    'LILATracts_Vehicle',
    'PovertyRate',
    'MedianFamilyIncome',
    'lawhitehalfshare'
]
corr_matrix = full_data[corr_features].dropna().corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap with Food Desert Status')
plt.tight_layout()
plt.show()

#%%

# Modeling Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

model_features = corr_features[1:]  # drop target
target = 'LILATracts_1And10'

model_data = full_data[model_features + [target]].dropna()
X = model_data[model_features]
y = model_data[target]

print("\nTarget Distribution:\n", y.value_counts(normalize=True))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
#Logistic Regression (Base)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("\n🔹 Base Logistic Regression")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

#%%
#Logistic Regression (Weighted)
logreg_weighted = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg_weighted.fit(X_train, y_train)
y_pred_w = logreg_weighted.predict(X_test)
print("\n🔹 Weighted Logistic Regression")
print(f"Accuracy: {accuracy_score(y_test,y_pred_w):.2f}")
print(classification_report(y_test, y_pred_w))

#%%
#GridSearchCV for Logistic Regression
params = {'C': [0.01, 0.1, 1, 10], 'class_weight': [None, 'balanced']}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid=params, cv=5, scoring='f1_macro')
grid.fit(X_train, y_train)
best_logreg = grid.best_estimator_
y_pred_best = best_logreg.predict(X_test)
print("\n🔹 GridSearch Best Logistic Regression")
print("Best Parameters:", grid.best_params_)
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.2f}")
print(classification_report(y_test, y_pred_best))

#%%
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(
    best_logreg, X_test, y_test, cmap='viridis', values_format='d'
).ax_.set_title("🔹 Best Logistic Regression - Confusion Matrix")
plt.show()

#%%
# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\n🔹 Random Forest Classifier")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print(classification_report(y_test, y_pred_rf))

#%%

# Feature Importance Plot
importances = pd.Series(rf.feature_importances_, index=model_features).sort_values(ascending=True)
plt.figure(figsize=(8, 5))
importances.plot(kind='barh')
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

#%%
# Summary Table
print("\n✅ Final Accuracy Summary")
print(f"Logistic (base):          {accuracy_score(y_test, y_pred):.2f}")
print(f"Logistic (weighted):      {accuracy_score(y_test, y_pred_w):.2f}")
print(f"Logistic (GridSearch):    {accuracy_score(y_test, y_pred_best):.2f}")
print(f"Random Forest:            {accuracy_score(y_test, y_pred_rf):.2f}")



#%%
''' DASH FRAMEWORK'''

#%%
print(full_data.columns.tolist())
#%%

# Grouped DataFrame for map
map_df = df.groupby('State_x').agg(
    total_tracts=('LILATracts_1And10', 'count'),
    food_deserts=('LILATracts_1And10', 'sum')
).reset_index()

map_df['desert_percent'] = round((map_df['food_deserts'] / map_df['total_tracts']) * 100, 2)

# Rename for mapping
map_df.rename(columns={'State_x': 'state'}, inplace=True)

#%%
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Load your data
df = full_data.copy()

# ============================
# Prep data for Choropleth Map
# ============================

map_df = df.groupby('State_x').agg(
    total_tracts=('LILATracts_1And10', 'count'),
    food_deserts=('LILATracts_1And10', 'sum')
).reset_index()
map_df['desert_percent'] = round((map_df['food_deserts'] / map_df['total_tracts']) * 100, 2)
map_df.rename(columns={'State_x': 'state'}, inplace=True)

us_state_abbrev = {  # same as before
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}
map_df['state'] = map_df['state'].map(us_state_abbrev)

# ============================
# Launch App
# ============================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "US Food Deserts Dashboard"

app.layout = html.Div([
    html.H2("US Food Deserts Dashboard", style={
        "textAlign": "center", "marginBottom": "30px", "fontSize": "30px", "fontFamily": "Arial"
    }),
    dcc.Tabs(id='tabs', value='tab-map', children=[
        dcc.Tab(label='🗺️ Geographic Map', value='tab-map'),
        dcc.Tab(label='📍 State Summary', value='tab-summary'),
        dcc.Tab(label='📊 Visualize Relationships', value='tab-visual')
    ]),
    html.Div(id='tab-content', style={"padding": "30px"})
])

# ============================
# Tab Switching Callback
# ============================
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'tab-map':
        fig = px.choropleth(
            map_df,
            locations='state',
            locationmode='USA-states',
            color='desert_percent',
            color_continuous_scale='YlOrRd',
            range_color=(0, map_df['desert_percent'].max()),
            scope='usa',
            labels={'desert_percent': '% Food Deserts'},
            hover_name='state',
            hover_data={'desert_percent': True, 'total_tracts': True, 'food_deserts': True, 'state': False},
        )
        fig.update_layout(
            geo=dict(bgcolor='white'),
            title='Choropleth Map: % of Census Tracts Classified as Food Deserts',
            title_font=dict(size=20, family='Arial', color='black'),
            margin={"r": 0, "t": 60, "l": 0, "b": 0},
            height=600
        )

        return html.Div([
            html.H3("Choropleth Map of Food Deserts", style={"marginBottom": "20px", "textAlign": "center"}),
            dcc.Graph(figure=fig)
        ])

    elif tab == 'tab-summary':
        return html.Div([
            html.H3("State-wise Food Desert Summary", style={"textAlign": "center", "marginBottom": "30px"}),
            dcc.Dropdown(
                id='state-summary-dropdown',
                options=[{'label': s, 'value': s} for s in sorted(df['State_x'].dropna().unique())],
                value='Alabama'
            ),
            html.Div(id='state-summary-output', style={"textAlign": "center"})
        ])

    elif tab == 'tab-visual':
        chart_types = ['Scatter', 'Box', 'Bar', 'Histogram']
        return html.Div([
            html.H3("Explore Relationships Between Variables", style={"marginBottom": "20px"}),
            html.Label("Chart type:", style={"fontSize": "16px"}),
            dcc.Dropdown(id='chart-type', options=[{'label': c, 'value': c.lower()} for c in chart_types],
                         value='scatter'),
            html.Br(),

            html.Label("Select X variable:", style={"fontSize": "16px"}),
            dcc.Dropdown(id='x-var'),

            html.Br(),

            html.Label("Select Y variable:", style={"fontSize": "16px"}),
            dcc.Dropdown(id='y-var'),

            html.Br(),
            dcc.Graph(id='custom-graph')
        ])


# ============================
# State Summary Callback
# ============================
@app.callback(Output('state-summary-output', 'children'), Input('state-summary-dropdown', 'value'))
def update_state_summary(state):
    if state is None:
        return html.Div("Please select a state.")
    state_data = df[df['State_x'] == state]
    
    total = state_data.shape[0]
    deserts = state_data['LILATracts_1And10'].sum()
    percent = round((deserts / total) * 100, 2) if total > 0 else 0
    
    return html.Div([
        html.Div([
            html.H4(f"🌍 {percent}% of census tracts in {state} are food deserts.", style={
                'fontWeight': 'bold', 'fontSize': '20px', 'marginBottom': '10px'
            }),
            html.P(f"🗺️ Total census tracts: {total}", style={'fontSize': '18px'}),
            html.P(f"🌾 Number of food desert tracts: {deserts}", style={'fontSize': '18px'})
        ], style={
            'border': '1px solid #ccc',
            'padding': '30px',
            'borderRadius': '12px',
            'backgroundColor': '#f9f9f9',
            'width': '50%',
            'margin': 'auto',
            'marginTop': '30px'
        })
    ])

# ============================
# Chart Options Logic
# ============================
@app.callback(
    [Output('x-var', 'options'),
     Output('y-var', 'options'),
     Output('x-var', 'value'),
     Output('y-var', 'value')],
    Input('chart-type', 'value')
)
def update_variable_options(chart_type):
    categorical = ['Urban', 'State_x', 'LowIncomeTracts', 'LILATracts_1And10']
    numeric = ['PovertyRate', 'MedianFamilyIncome', 'lasnap10share', 'lalowihalfshare',
               'PCT_OBESE_ADULTS17', 'GROCPTH16', 'PCT_SNAP17', 'MEDHHINC15']

    if chart_type == 'scatter':
        opts = [{'label': col, 'value': col} for col in numeric]
        return opts, opts, numeric[0], numeric[1]
    elif chart_type in ['box', 'bar']:
        x_opts = [{'label': col, 'value': col} for col in categorical]
        y_opts = [{'label': col, 'value': col} for col in numeric]
        return x_opts, y_opts, categorical[0], numeric[0]
    elif chart_type == 'histogram':
        opts = [{'label': col, 'value': col} for col in numeric]
        return opts, [], numeric[0], None
    else:
        return [], [], None, None

# ============================
# Chart Renderer
# ============================
@app.callback(
    Output('custom-graph', 'figure'),
    Input('x-var', 'value'),
    Input('y-var', 'value'),
    Input('chart-type', 'value'),
    prevent_initial_call=True
)
def update_custom_graph(x, y, chart):
    if chart == 'histogram':
        return px.histogram(df, x=x, nbins=40, color_discrete_sequence=['#6C3483'])
    elif chart == 'scatter':
        return px.scatter(df, x=x, y=y, color_discrete_sequence=['#FF5733'])
    elif chart == 'box':
        return px.box(df, x=x, y=y, color_discrete_sequence=['#2980B9'])
    elif chart == 'bar':
        grouped = df.groupby(x)[y].mean().reset_index()
        return px.bar(grouped, x=x, y=y, color_discrete_sequence=['#239B56'])
    return px.scatter()

# ============================
# Run App
# ============================
if __name__ == '__main__':
    app.run_server(debug=True)


#%%




#%%
