# Heart Disease Prediction with XGBoost

## Overview

This project implements a machine learning model to predict heart disease using the XGBoost algorithm. The dataset used is the Heart Disease UCI dataset, which contains various medical attributes to classify the presence and severity of heart disease. The project includes comprehensive data preprocessing, model training, evaluation, and sample predictions.

The notebook (`Code_XG_Boost.ipynb`) demonstrates the end-to-end process, from data acquisition to model deployment for predictions.

## Features

- **Data Acquisition**: Downloads the heart disease dataset from Kaggle.
- **Data Preprocessing**:
  - Handling missing values using iterative imputation for both continuous and categorical features.
  - Outlier detection and treatment using Z-score method.
  - Encoding categorical variables.
- **Exploratory Data Analysis (EDA)**: Visualizations for data distribution, missing values, correlations, and feature importance.
- **Model Training**: Uses XGBoost classifier with hyperparameter tuning.
- **Model Evaluation**: Includes accuracy, classification report, and confusion matrix.
- **Predictions**: Demonstrates predictions on sample patient data.

## Dataset

The dataset (`heart_disease_uci.csv`) is sourced from Kaggle (redwankarimsony/heart-disease-data). It includes features such as age, sex, chest pain type, blood pressure, cholesterol, etc., with the target variable `num` indicating heart disease severity (0-4).

## Installation and Setup

### Prerequisites

- Python 3.7+
- Kaggle API (for data download)
- Jupyter Notebook or JupyterLab

### Dependencies

Install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy kaggle
```

### Data Download

1. Set up Kaggle API:
   - Install Kaggle CLI: `pip install kaggle`
   - Download your Kaggle API token from your account settings.
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows).

2. Run the notebook to download and unzip the dataset:
   - The notebook includes commands to download `heart-disease-data.zip` and extract it to a `dataset` folder.

### Running the Notebook

1. Clone or download this repository.
2. Navigate to the `Code_XG_Boost` directory.
3. Open `Code_XG_Boost.ipynb` in Jupyter Notebook.
4. Run the cells sequentially to execute the analysis and model training.

## Usage

### Data Preprocessing

- The notebook handles missing values by imputing continuous features with linear regression and categorical features with logistic regression.
- Outliers are detected and removed using Z-score thresholding.

### Model Training

- Splits data into training and testing sets (80/20 split with stratification).
- Trains an XGBoost classifier with specified hyperparameters (e.g., learning rate, max depth).
- Evaluates the model on the test set.

### Making Predictions

- The notebook includes sample patient data for prediction.
- Use the trained model to predict heart disease severity for new data.

Example prediction output:

| id   | age | sex | dataset | cp | trestbps | chol | fbs | restecg | thalch | exang | oldpeak | slope | ca | thal | Prediction |
|------|-----|-----|---------|----|----------|------|-----|---------|--------|-------|---------|-------|----|------|------------|
| 999  | 45  | 1   | 1       | 0  | 130      | 233  | True| 0       | 150    | False | 2.3     | 0     | 0  | 0    | Pas de maladie : 0 |
| ...  | ... | ... | ...     | ...| ...      | ...  | ... | ...     | ...    | ...   | ...     | ...   | ...| ...  | ...        |

## Results

- **Accuracy**: The model achieves an accuracy of approximately 85-90% (based on the notebook's output; may vary with data).
- **Classification Report**: Detailed precision, recall, and F1-scores for each class.
- **Confusion Matrix**: Visual representation of true positives, false positives, etc.
- **Feature Importance**: Highlights key features like `thal`, `ca`, `oldpeak`, etc.

## Hyperparameters

The XGBoost model uses the following key hyperparameters:
- `learning_rate`: 0.01
- `n_estimators`: 20
- `max_depth`: 3
- `min_child_weight`: 2
- `random_state`: 30

These can be tuned further using RandomizedSearchCV for better performance.

## Contributing

Feel free to fork this repository and submit pull requests for improvements, such as additional feature engineering, hyperparameter optimization, or model comparisons.

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- Dataset: [Heart Disease Data on Kaggle](https://www.kaggle.com/redwankarimsony/heart-disease-data)
- Libraries: XGBoost, Scikit-learn, Pandas, Matplotlib, Seaborn
