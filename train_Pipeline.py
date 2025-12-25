import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib

#Loading the dataset
df = pd.read_csv("Telco-Churn.csv")

#print First 5 rows of datset
print(df.head())
print("Dataset shape: ", df.shape)

print("churn Distribution: ", df['Churn'].value_counts())

#Data Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#Handle missing values
df.dropna(inplace=True)

#Change target col to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("After cleaning: ", df.isnull().sum())
print("Cleaned dataset shape: ", df.shape)

#Seperate the target col
x = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

print("Features shape: ", x.shape)
print("Target shape: ", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
print("Training Data: ", x_train.shape)
print("Testing data: ", x_test.shape)

#get col types
num_col = x.select_dtypes(include=['int64', 'float64']).columns
cat_col = x.select_dtypes(include=['object']).columns

print("Numerical columns:", list(num_col))
print("Categorical columns:", list(cat_col))

#Column Transformer
preprocessor = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), num_col),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_col)
    ]
)

#make ML Pipeline
pipeline = Pipeline(
    steps = [
        ('preprocessing', preprocessor),
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ]
)

#applying GridSearchCV
para  = [
    {
        'model': [LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)],
        'model__C': [0.01, 0.1, 1, 10, 100]
    },
    {
        'model': [RandomForestClassifier(random_state=42)],
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 5, 10]
    }]
grid = GridSearchCV(pipeline, para, scoring='f1', cv=5)

grid.fit(x_train, y_train)
best_model = grid.best_estimator_
y_pred_best = best_model.predict(x_test)
print("Classification Report(bestModel): ", classification_report(y_test, y_pred_best))

#Saving the trained model
joblib.dump(best_model, "churnModel.pkl")
print("Model saved successfully")
