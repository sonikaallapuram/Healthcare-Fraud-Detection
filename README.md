# 🏥 Healthcare Insurance Fraud Detection

## 📌 Project Overview
An end-to-end Data Science project that uses Machine Learning to automatically detect fraudulent healthcare insurance claims. This project analyzes over 558,000 real world healthcare claims using XGBoost classifier with SMOTE balancing and SHAP explainability.

## 🎯 Problem Statement
Healthcare insurance fraud costs billions of dollars annually, placing a massive burden on patients, insurers, and the healthcare system. Fraudulent claims continue to rise, making it increasingly difficult for insurance companies to identify them manually. This project uses Machine Learning to automatically detect fraudulent claims and provide actionable business insights.

⚠️ Note: This project is a work in progress. Some features are incomplete, and bugs may exist. I am continuously learning and improving the project over time.

## 📸 Screenshots

**. Overview**
<img width="1316" height="610" alt="image" src="https://github.com/user-attachments/assets/38af8815-22dd-4f2c-bed3-ff27eab99f5f" />

**.Fraud Prediction**

<img width="1323" height="588" alt="image" src="https://github.com/user-attachments/assets/ffdefb28-2c56-4e4d-9130-5f1491196afa" />

## 📊 Dataset
- Source: Kaggle - Healthcare Provider Fraud Detection Analysis
- Total Records: 558,211 claims
- Files Used:
  - Train_Beneficiarydata.csv
  - Train_Inpatientdata.csv
  - Train_Outpatientdata.csv
  - Train_data.csv

## 🛠️ Tech Stack
- Language: Python
- Data Analysis: Pandas, Numpy
- Visualization: Matplotlib, Seaborn
- Machine Learning: Scikit-learn, XGBoost
- Explainability: SHAP
- Class Balancing: SMOTE
- Dashboard: Streamlit
- Deployment: Streamlit Cloud
- Version Control: GitHub

## 📁 Project Structure
Healthcare-Fraud-Detection/
├── EDA.ipynb
├── DataCleaning.ipynb
├── ModelBuilding.ipynb
├── app.py
├── cleaned_data.csv
├── fraud_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
    ├── screenshot_overview.png
    ├── screenshot_analysis.png
    ├── screenshot_model.png
    └── screenshot_prediction.png

## 🔄 Project Pipeline

### Data Collection and EDA
- Loaded and merged 4 datasets into one
- Performed Exploratory Data Analysis
- Visualized fraud distribution and claim patterns
- Identified key fraud indicators

### Data Cleaning and Feature Engineering
- Removed duplicates and handled missing values
- Converted date columns to proper format
- Created new meaningful features:
  - Age from date of birth
  - Claim Duration from start and end dates
  - Hospital Stay Days from admission and discharge dates
  - Is Deceased from date of death
- Encoded categorical variables using LabelEncoder

### Model Building and Evaluation
- Applied SMOTE to handle class imbalance
- Split data 80% training and 20% testing
- Built and compared two models:
  - Random Forest Classifier
  - XGBoost Classifier
- Evaluated using:
  - Accuracy Score
  - Confusion Matrix
  - ROC Curve and AUC Score
  - SHAP Values for Explainability
- Saved best model using pickle

### Dashboard
- Built interactive 4 page Streamlit dashboard
- Project Overview with key metrics
- Data Analysis with visualizations
- Model Performance with evaluation charts
- Real time Fraud Prediction

## 📈 Model Results

Model          | Accuracy | AUC Score
---------------|----------|----------
Random Forest  | 83.73%   | 0.94
XGBoost        | 90.28%   | 0.97

## 📊 Key Findings
- High claim amounts are strongest indicator of fraud
- Deceased patient claims are clear fraud indicators
- Short hospital stays with high bills indicate fraud
- XGBoost significantly outperforms Random Forest
- SMOTE balancing improved model performance

## 🚀 How to Run Locally

### 1. Clone the repository
git clone https://github.com/sonikaallapuram/Healthcare-Fraud-Detection.git
cd Healthcare-Fraud-Detection

### 2. Install dependencies
pip install -r requirements.txt

### 3. Download Dataset from Kaggle
- Go to www.kaggle.com
- Search Healthcare Provider Fraud Detection Analysis
- Download and place CSV files in project folder

### 4. Run Notebooks
jupyter notebook EDA.ipynb
jupyter notebook DataCleaning.ipynb
jupyter notebook ModelBuilding.ipynb

### 5. Run Streamlit Dashboard
streamlit run app.py

## 🌐 Live Demo
[🚀 Click here to view live dashboard](https://healthcare-fraud-detection-sonika.streamlit.app/)

🤝 Contributing Suggestions and feedback are always welcome:

📄 License This project is licensed under the MIT License.

💡 This project reflects my journey as a student developer aspiring to build real-world applications. While not yet perfect, I’m committed to learning and improving it step by step.
