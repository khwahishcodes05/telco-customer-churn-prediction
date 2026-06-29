# # ==========================================
# TELCO CUSTOMER CHURN PREDICTION PROJECT
# ==========================================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ==========================================
# Import Required Libraries
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Model Evaluation
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("/content/drive/MyDrive/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ==========================================
# Basic Data Exploration
# ==========================================

print(df.head())          # First 5 rows
print(df.info())          # Dataset information
print(df.describe())      # Statistical summary
print(df.shape)           # Rows & Columns
print(df.size)            # Total values
print(df.isnull().sum())  # Missing values
print(df.duplicated())    # Duplicate rows

# ==========================================
# Target Variable Distribution
# ==========================================

print(df["Churn"].value_counts(normalize=True))

ax = sns.countplot(x=df["Churn"], data=df)

for container in ax.containers:
    ax.bar_label(container)

plt.show()

# ==========================================
# Data Cleaning
# ==========================================

# Create a copy of original dataset
df2 = df.copy()

# Convert TotalCharges into numeric datatype
df2.TotalCharges = pd.to_numeric(df2.TotalCharges, errors="coerce")

# Check missing values
print(df2["TotalCharges"].isnull().sum())

# Remove missing rows
df2.dropna(subset=["TotalCharges"], inplace=True)

print(df2.isnull().sum())

# ==========================================
# Feature Engineering
# ==========================================

# Create tenure groups
labels = ["{0}-{1}".format(i, i+11) for i in range(1,72,12)]

df2["tenure_group"] = pd.cut(
    df2["tenure"],
    range(1,80,12),
    right=False,
    labels=labels
)

print(df2["tenure_group"].value_counts())

# Drop unnecessary columns
df2.drop(columns=["customerID","tenure"], inplace=True)

# Convert target variable into binary
df2["Churn"] = np.where(df2["Churn"]=="Yes",1,0)

# One Hot Encoding
df2_dummy = pd.get_dummies(df2, drop_first=True)
df2_dummy = df2_dummy.astype(int)

# ==========================================
# Data Visualization
# ==========================================

# Countplots for categorical variables
for i, predictor in enumerate(
        df2.drop(columns=["TotalCharges","MonthlyCharges","Churn"])
):
    plt.figure(i)
    sns.countplot(
        x=predictor,
        data=df2,
        hue="Churn",
        palette="viridis"
    )

plt.show()

# Monthly Charges vs Churn
plt.figure(figsize=(12,12))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df2)
plt.show()

# Total Charges vs Churn
plt.figure(figsize=(10,12))
sns.boxplot(x="Churn", y="TotalCharges", data=df2)
plt.show()

# Contract Type vs Churn
plt.figure(figsize=(8,12))
sns.countplot(
    x="Contract",
    hue="Churn",
    data=df2
)
plt.title("Churn by Contract Type")
plt.show()

# Scatter Plot
plt.figure(figsize=(12,12))
sns.scatterplot(
    x="TotalCharges",
    y="MonthlyCharges",
    data=df2,
    hue="Churn"
)
plt.show()

# KDE Plot - Monthly Charges
plt.figure(figsize=(12,12))

sns.kdeplot(
    data=df2[df2["Churn"]==0],
    x="MonthlyCharges",
    fill=True,
    label="No Churn"
)

sns.kdeplot(
    data=df2[df2["Churn"]==1],
    x="MonthlyCharges",
    fill=True,
    label="Churn"
)

plt.legend()
plt.show()

# KDE Plot - Total Charges
plt.figure(figsize=(12,12))

sns.kdeplot(
    data=df2[df2["Churn"]==0],
    x="TotalCharges",
    fill=True,
    label="No Churn"
)

sns.kdeplot(
    data=df2[df2["Churn"]==1],
    x="TotalCharges",
    fill=True,
    label="Churn"
)

plt.legend()
plt.show()

# ==========================================
# Correlation Analysis
# ==========================================

corr = df2_dummy.corr()["Churn"].abs().sort_values(ascending=False)

corr.plot(kind="bar", figsize=(10,5))
plt.title("Feature Importance")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df2_dummy.corr(), cmap="coolwarm")
plt.show()

# ==========================================
# Train-Test Split
# ==========================================

X = df2_dummy.drop("Churn", axis=1)
y = df2_dummy["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Machine Learning Models
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss")
}

# ==========================================
# Model Training & Evaluation
# ==========================================

result = []

for name, model in models.items():

    # Apply scaling only where required
    if name in ["Logistic Regression","Naive Bayes","KNN","SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    result.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test,y_pred),4),
        "Precision": round(precision_score(y_test,y_pred),4),
        "Recall": round(recall_score(y_test,y_pred),4),
        "F1 Score": round(f1_score(y_test,y_pred),4)
    })

    print(f"\n{name}")
    print(classification_report(y_test,y_pred))

# ==========================================
# Final Model Comparison
# ==========================================

results_df = pd.DataFrame(result)

print(results_df.sort_values(by="Accuracy", ascending=False))