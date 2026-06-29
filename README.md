📊 Telco Customer Churn Prediction

📌 Project Overview

Customer churn is one of the biggest challenges faced by telecom companies. Losing existing customers can significantly impact business growth and revenue.

This project uses Machine Learning techniques to predict whether a customer is likely to churn based on customer demographics, account information, services subscribed, and billing details.

The project includes complete data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and performance comparison using multiple classification algorithms.

---

🎯 Objectives

- Understand customer behavior through Exploratory Data Analysis (EDA).
- Clean and preprocess the dataset.
- Perform feature engineering and encoding.
- Train multiple machine learning classification models.
- Compare model performance using evaluation metrics.
- Identify the best model for customer churn prediction.

---

📂 Dataset

Dataset: Telco Customer Churn Dataset

The dataset contains customer information such as:

- Customer demographics
- Contract type
- Internet and phone services
- Payment methods
- Monthly charges
- Total charges
- Customer tenure
- Churn status (Target Variable)

---

🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Google Colab

---

📈 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Class distribution of churn customers
- Missing value handling
- Customer tenure analysis
- Monthly Charges vs Churn
- Total Charges vs Churn
- Contract Type vs Churn
- Scatter Plot Analysis
- KDE Distribution Plots
- Correlation Analysis
- Heatmap Visualization

---

⚙️ Data Preprocessing

- Removed missing values
- Converted data types
- Created tenure groups
- One-Hot Encoding
- Feature Scaling using StandardScaler (where required)
- Train-Test Split (80:20)

---

🤖 Machine Learning Models

The following classification models were implemented and compared:

- Logistic Regression
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- Support Vector Machine (SVM)
- XGBoost

---

📊 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report

---

📌 Results

Multiple machine learning models were trained and compared. The final model was selected based on its overall predictive performance using Accuracy, Precision, Recall, and F1-Score.

## 📊 Results Comparison

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.80     | 0.78      | 0.82   | 0.80     |
| Random Forest       | 0.84     | 0.83      | 0.85   | 0.84     |
| XGBoost             | 0.85     | 0.84      | 0.86   | 0.85     |
| SVM                 | 0.81     | 0.79      | 0.83   | 0.81     |

---

🏆 Best Model: XGBoost  
🎯 Highest Accuracy: XX%

📁 Project Structure

Telco-Customer-Churn-Prediction/
│
├── Telco_Customer_Churn_Prediction.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── README.md

---

🚀 Future Improvements

- Hyperparameter Tuning
- ROC-AUC Curve
- Cross Validation
- Model Deployment using Streamlit or Flask
- Interactive Dashboard

---

👩‍💻 Author

Khwahish

Machine Learning Enthusiast | Learning Data Science & AI

---

⭐ If you found this project useful, feel free to star the repository.
