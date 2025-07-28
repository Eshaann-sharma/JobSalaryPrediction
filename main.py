import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Load Data ---
data = pd.read_csv("Salary Data.csv", on_bad_lines='skip')
print("Data loaded. First 5 rows:\n", data.head())

# --- 2. Initial Data Exploration ---
print("\nData info:")
print(data.info())
print("\nMissing values per column:")
print(data.isnull().sum())
print("\nBasic statistics:")
print(data.describe())

# Visualize Salary distribution
plt.figure(figsize=(8,5))
sns.histplot(data['Salary'], bins=30, kde=True)
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Count')
plt.show()

# Visualize Years of Experience distribution
plt.figure(figsize=(8,5))
sns.histplot(data['Years of Experience'], bins=20, kde=True)
plt.title('Years of Experience Distribution')
plt.xlabel('Years of Experience')
plt.ylabel('Count')
plt.show()

# Salary by Education Level boxplot
plt.figure(figsize=(8,6))
sns.boxplot(x='Education Level', y='Salary', data=data)
plt.title('Salary by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Salary')
plt.show()

# Salary by Job Title (top 10 frequent jobs)
top_jobs = data['Job Title'].value_counts().nlargest(10).index
plt.figure(figsize=(10,6))
sns.boxplot(x='Salary', y='Job Title', data=data[data['Job Title'].isin(top_jobs)])
plt.title('Salary Distribution by Top 10 Job Titles')
plt.xlabel('Salary')
plt.ylabel('Job Title')
plt.show()

# --- 3. Data Cleaning ---
# Drop rows with missing Salary or Years of Experience
data_clean = data.dropna(subset=['Salary', 'Years of Experience'])

# Convert Salary and Years of Experience to numeric (force errors to NaN then drop)
data_clean['Salary'] = pd.to_numeric(data_clean['Salary'], errors='coerce')
data_clean['Years of Experience'] = pd.to_numeric(data_clean['Years of Experience'], errors='coerce')
data_clean = data_clean.dropna(subset=['Salary', 'Years of Experience'])

# --- 4. Encoding categorical variables ---
categorical_cols = ['Gender', 'Education Level', 'Job Title']
data_encoded = pd.get_dummies(data_clean, columns=categorical_cols, drop_first=True)

# --- 5. Features and Target ---
X = data_encoded.drop('Salary', axis=1)
y = data_encoded['Salary']

# --- 6. Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 7. Model Training ---
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, objective='reg:squarederror')
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'model': model, 'mse': mse, 'r2': r2}
    print(f"{name} -- MSE: {mse:.2f}, R2: {r2:.3f}")

# --- 8. Visualize Model Performance ---
model_names = list(results.keys())
mses = [results[m]['mse'] for m in model_names]

plt.figure(figsize=(8,5))
sns.barplot(x=model_names, y=mses)
plt.title('Model Mean Squared Error (Lower is better)')
plt.ylabel('MSE')
plt.show()

# --- 9. Residual Analysis for Best Model ---
best_model_name = min(results, key=lambda k: results[k]['mse'])
best_model = results[best_model_name]['model']
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred_best)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title(f'Residual Analysis: {best_model_name}')
plt.show()

residuals = y_test - y_pred_best
plt.figure(figsize=(8,5))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals (Actual - Predicted)')
plt.title(f'Residual Distribution: {best_model_name}')
plt.show()

# --- 10. Feature Importance (for tree-based models) ---
def plot_feature_importance(model, X, title):
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        feat_imp = pd.Series(fi, index=X.columns).sort_values(ascending=False)[:15]
        plt.figure(figsize=(10,6))
        sns.barplot(x=feat_imp.values, y=feat_imp.index)
        plt.title(title)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.show()

print(f"\nFeature Importance for {best_model_name}:")
plot_feature_importance(best_model, X_train, f'Feature Importance - {best_model_name}')

# --- 11. Sample Prediction with input explanation ---
print("\nSample test input and prediction:")
sample_index = 0
sample = X_test.iloc[[sample_index]]
print("Input features:\n", sample.T)

for name, model in models.items():
    pred = model.predict(sample)[0]
    print(f"{name} predicts salary: ${pred:,.2f}")

