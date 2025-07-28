# Job Salary Prediction

## Dataset
Source: Kaggle Job Salary Prediction Dataset

Fields: Age, Gender, Education Level, Job Title, Years of Experience, Salary


## Installation
Clone this repository:

```
git clone https://github.com/Eshaann-sharma/JobSalaryPrediction.git
```
```
cd job-salary-prediction
```

Install dependencies:

```
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```
On macOS, for XGBoost runtime:

```
brew install libomp
```

## Results & Visualizations
Model Comparison:
The script compares Linear Regression, Random Forest, and XGBoost, reporting MSE and R² for each.
Data Insights:
Rich visualizations: salary histograms, relation plots by education/job title, feature importances, and residuals.
Sample Output:
Predictions for actual unseen samples with explanation of input features.

## References
Kaggle Job Salary Prediction

scikit-learn documentation

XGBoost documentation

pandas, numpy, matplotlib, seaborn
