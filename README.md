# XGBoost Project - Heart Disease Prediction

## 📋 Overview

This project performs **predictive analysis and classification of heart disease** using the **XGBoost** algorithm. The objective is to develop a machine learning model capable of predicting the presence of heart disease in patients based on various medical characteristics.

### Dataset
- **Source** : [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- **Number of Samples** : 920 patients
- **Number of Features** : 16 variables (after preparation)
- **Target** : Variable `num` (0 = no disease, 1-4 = degrees of heart disease)

---

## 🗂️ Project Structure

```
Projet_XGBoost/
├── Code_XG_Boost/
│   ├── Code_XG_Boost.ipynb           # Jupyter Notebook with complete analysis
│   └── heart_disease_uci.csv         # Raw data
├── Rapport_XGBoost.pdf               # Detailed project report
├── XG_Boost_ML_project.pptx         # PowerPoint presentation
└── README.md                          # This file
```

---

## 🎯 Project Steps

### 1. **Data Import and Loading**
   - Dataset download from Kaggle
   - Data loading with Pandas
   - Initial data exploration

**Libraries Used :**
```python
- pandas (pd)
- numpy (np)
- matplotlib.pyplot
- seaborn (sns)
- scikit-learn
- xgboost
```

### 2. **Exploratory Data Analysis (EDA)**
   - Visualization of target variable distribution
   - Analysis of missing values
   - Identification of columns with missing data :
     - `trestbps`, `chol`, `fbs`, `restecg`, `thalch`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

### 3. **Data Preprocessing**

#### 3.1 Missing Values Treatment
- **Continuous Columns** (numerical) : Iterative imputation with linear regression
  - Examples : `oldpeak`, `thalch`, `chol`, `trestbps`
  - Model : `LinearRegression()` + `IterativeImputer`
  
- **Categorical Columns** : Iterative imputation with logistic regression
  - Examples : `thal`, `ca`, `slope`, `exang`, `restecg`, `fbs`
  - Model : `LogisticRegression(max_iter=1000)` + `IterativeImputer`
  - Accuracy rates displayed for each feature

#### 3.2 Outlier Detection and Treatment
- **Method** : Z-score with threshold = 3
- **Affected Columns** : 
  - `trestbps` (6 outliers removed)
  - `chol` (2 outliers removed)
  - `thalch` (1 outlier removed)
  - `oldpeak` (5 outliers removed)

### 4. **Data Encoding**
- **Categorical Variables Encoding** :
  - Identified Columns : `thal`, `ca`, `slope`, `exang`, `restecg`, `fbs`, `cp`, `sex`, `num`
  - Boolean Variables : `fbs`, `exang`
  - Use of `LabelEncoder` to transform categories into numerical values

### 5. **Normalization/Standardization**
- Scaling of numerical features
- Normalization of continuous columns to improve model convergence

### 6. **Train/Test Split**
- **Ratio** : 70% train, 30% test
- **Random State** : 42 (for reproducibility)

### 7. **XGBoost Model Training**
- **Algorithm** : `XGBClassifier` from XGBoost library
- **Hyperparameter Optimization** : `RandomizedSearchCV`
- **Performance Metrics** :
  - Accuracy
  - Classification Report
  - Confusion Matrix

### 8. **Results and Evaluation**
- Calculation of performance metrics
- Confusion matrix visualization
- Detailed classification report display

---

## 📊 Dataset Features

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Patient age | Numerical |
| `sex` | Patient gender (Male/Female) | Categorical |
| `cp` | Type of chest pain | Categorical |
| `trestbps` | Resting blood pressure | Numerical |
| `chol` | Cholesterol level | Numerical |
| `fbs` | Fasting blood sugar > 120 mg/dl | Boolean |
| `restecg` | Resting ECG results | Categorical |
| `thalch` | Maximum heart rate achieved | Numerical |
| `exang` | Exercise-induced angina | Boolean |
| `oldpeak` | ST depression induced by exercise | Numerical |
| `slope` | Slope of ST segment | Categorical |
| `ca` | Number of major vessels colored | Categorical |
| `thal` | Thalassemia | Categorical |
| `num` | **TARGET** : Presence of heart disease (0-4) | Categorical |

---

## 🔧 Installation and Configuration

### Prerequisites
- **Python** 3.7+
- **Jupyter Notebook** or **Google Colab**

### Installing Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy
```

### Running the Notebook
```bash
# On Google Colab
# 1. Open : https://colab.research.google.com/
# 2. Upload or download the notebook

# Locally
jupyter notebook Code_XG_Boost/Code_XG_Boost.ipynb
```

---

## 📈 Model Performance

The trained XGBoost model provides :
- **Accuracy** : Correct classification rate
- **Precision, Recall, F1-Score** : Detailed in classification report
- **Confusion Matrix** : Visualization of classification errors

---

## 📚 Additional Files

- **Rapport_XGBoost.pdf** : Detailed report with visualizations and conclusions
- **XG_Boost_ML_project.pptx** : PowerPoint presentation of the project

---

## 🎓 Key Concepts Used

1. **Machine Learning Classification** : Binary/multi-class prediction
2. **Feature Engineering** : Data preparation and transformation
3. **Data Imputation** : Handling missing values
4. **Outlier Detection** : Identification of anomalies
5. **XGBoost** : Gradient Boosting for classification
6. **Hyperparameter Tuning** : Optimization via RandomizedSearchCV
7. **Cross-Validation** : Robust model evaluation

---

## 👨‍💼 Author

Project created to analyze and predict heart disease using modern Machine Learning techniques.

---

## 📝 License

Dataset License : copyright-authors (source : Kaggle)

---

## 🤝 Contributing

Feel free to suggest improvements or modifications to this project.

---

## ⚠️ Important Notes

- The model is trained on normalized UCI Heart Disease data
- Performance may vary depending on input data
- Imputation of missing values may impact final results
- Removal of outliers may reduce the dataset size
- Always validate results on an independent test set

---

**Last Updated** : 2025
