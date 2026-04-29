import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load the processed profiles
df = pd.read_csv('merged_fintech_data.csv')

# 2. Define Features (X) and Target (y)
features = ['avg_spend', 'total_spend', 'trans_count', 'city_pop', 'fraud_history']
X = df[features]
y = df['target']

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Handle Imbalance
# We calculate the ratio of (Active / Churned) to tell the model to focus
ratio = 908 / 30 

print(f"Training XGBoost with Weight Ratio: {ratio:.2f}")

# 5. Initialize and Train XGBoost
# scale_pos_weight is the secret sauce for imbalanced FinTech data
# Instead of using the full 'ratio', we use a smaller multiplier
# This balances the model so it doesn't call everyone a 'Churner'
model = XGBClassifier(
    n_estimators=100,        # More trees now that we have more data
    max_depth=5,             # Deeper trees can find more complex patterns
    learning_rate=0.1,
    scale_pos_weight=4,      # Balanced for typical bank churn (approx 20-25% churn)
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("\n--- Improved Model Performance Report ---")
print(classification_report(y_test, y_pred))

# 7. Save
joblib.dump(model, 'churn_model.pkl')
print("\n✅ Success: Model saved!")