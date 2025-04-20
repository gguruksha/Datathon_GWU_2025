# 🥇 GWU Datathon 2025 — Food Desert Analysis & Prediction

This project was developed as a solo submission for the 2025 George Washington University Datathon and secured 🥈2nd place. The goal was to analyze U.S. census tract–level data to identify food deserts using machine learning, and present actionable insights through an interactive, equity-driven dashboard.

---

## 📌 Problem Statement

Food deserts — areas with limited access to affordable and nutritious food — have serious implications for public health and social equity. Using tract-level USDA data, this project aims to:

- Identify key drivers of food inaccessibility  
- Predict food desert likelihood using machine learning  
- Build an interactive dashboard to support data-driven policymaking

---

## 📊 Dataset

- 📁 USDA Food Access Research Atlas  
- 📁 USDA Food Environment Atlas  
- ~70,000 census tracts, 100+ features  
- Cleaned, merged, and preprocessed to ensure quality insights

---

## 🧠 Methodology

### 🔍 Exploratory Data Analysis (EDA)
- Univariate & bivariate stats
- Correlation heatmaps
- Grouped means and boxplots by food desert status

### ⚙️ Feature Engineering
- Handling multicollinearity  
- Normalization and encoding  
- Custom metrics like “SNAP households per grocery store”

### 🧪 Modeling
- Logistic Regression (base and weighted)
- Hyperparameter tuning with GridSearchCV  
- Random Forest Classifier for non-linear patterns  
- ROC-AUC, Precision-Recall, and Feature Importance evaluation

---

## 🖥️ Interactive Dashboard

Built using Python Dash:

- 🗺️ Tab 1: U.S. map showing food desert tracts  
- 📊 Tab 2: State-level summaries via dropdown  
- 📈 Tab 3: Custom visualization — choose any two variables + chart type  
- ✨ Dark mode, tooltips, exportable charts, and sidebar metadata

---

## 📈 Key Results

- Random Forest achieved ~80% accuracy in identifying food deserts  
- Key predictors: SNAP participation, poverty rate, vehicle access  
- Custom metrics revealed deeper disparities than binary labels alone  
- Interactive dashboard enables deeper exploration by policy teams and researchers

---

## 🧩 Future Work

- Integrate temporal changes in food accessibility  
- Deploy on cloud with real-time updates  
- Suggest targeted interventions using model outputs  
- Incorporate geospatial overlays and clustering

---

## 👩‍💻 Author

Built and maintained by **Guruksha Gurnani**  
📍 Master’s in Data Science, George Washington University  
🌐 GitHub: [@gguruksha](https://github.com/gguruksha)  
📫 LinkedIn: [linkedin.com/in/gurukshagurnani](https://www.linkedin.com/in/gurukshagurnani/)


---
