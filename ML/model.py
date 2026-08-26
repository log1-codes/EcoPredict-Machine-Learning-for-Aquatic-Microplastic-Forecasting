import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("[1/5] Loading CalCOFI dataset...")

columns_to_use = ['Depthm', 'Salnty', 'O2ml_L', 'T_degC']

try:
    df = pd.read_csv('bottle.csv', usecols=columns_to_use)
except FileNotFoundError:
    print("Error: 'bottle.csv' not found. Ensure the dataset is in the working directory.")
    exit()

print(f"Original dataset shape: {df.shape}")

df_clean = df.dropna().copy()
print(f"Cleaned dataset shape: {df_clean.shape}")

if len(df_clean) > 60000:
    df_sample = df_clean.sample(n=60000, random_state=42)
else:
    df_sample = df_clean

X = df_sample[['Depthm', 'Salnty', 'O2ml_L']]
y = df_sample['T_degC']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')

print("\n[2/5] Training models for comparative analysis...")

models = {
    "Multiple Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
}

results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results.append({
        "Model": name,
        "R2 Score": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "Trained_Model": model
    })

print("\n[3/5] Evaluation Complete. Summary Table:")
results_df = pd.DataFrame(results).drop(columns=['Trained_Model'])
print(results_df.to_string(index=False))



print("\n[4/5] Generating publication charts...")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Chart 1: R2 Score Comparison
sns.barplot(x="Model", y="R2 Score", data=results_df, ax=axes[0], palette="Blues_r")
axes[0].set_title("Algorithm Comparison: R² Score (Higher is Better)", fontsize=12)
axes[0].set_ylim(0, 1.0)
axes[0].tick_params(axis='x', rotation=20)

# Chart 2: RMSE Comparison
sns.barplot(x="Model", y="RMSE", data=results_df, ax=axes[1], palette="Reds_r")
axes[1].set_title("Algorithm Comparison: Root Mean Squared Error (Lower is Better)", fontsize=12)
axes[1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig("model_comparison_metrics.png", dpi=300)
print("Saved comparison chart as 'model_comparison_metrics.png'")

print("\n[5/5] Exporting best model...")

best_model_info = max(results, key=lambda x: x['R2 Score'])
best_model = best_model_info['Trained_Model']
joblib.dump(best_model, 'best_ecopredict_model.pkl')

print(f"Top Model: {best_model_info['Model']} (R²: {best_model_info['R2 Score']})")
print("Saved best model artifact as 'best_ecopredict_model.pkl'")