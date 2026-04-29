import joblib
import pandas as pd
import numpy as np

# 1. Load the "Brain" (Model)
try:
    model = joblib.load('churn_model.pkl')
    # These must match the order you used during training
    features = ['avg_spend', 'total_spend', 'trans_count', 'city_pop', 'fraud_history']
    print("✅ Model loaded successfully.\n")
except:
    print("❌ Error: churn_model.pkl not found. Run train_model.py first.")
    exit()

def evaluate_customer(data_list, profile_name):
    # Convert list to DataFrame for the model
    df = pd.DataFrame([data_list], columns=features)
    
    # Get Probability (0 to 1)
    probability = model.predict_proba(df)[0][1]
    
    # --- UPDATE THIS PART ---
    # We use 0.15 (15%) because in banking, even a small risk 
    # is worth investigating.
    custom_threshold = 0.15 
    
    if probability > custom_threshold:
        status = "🚨 CHURN RISK"
    else:
        status = "✅ LOYAL"
    # -------------------------
    
    print(f"--- Profile: {profile_name} ---")
    print(f"Features: {data_list}")
    print(f"Result  : {status}")
    print(f"Confidence Score: {probability:.2%}\n")

# 2. RUN DIFFERENT DATASETS (Scenarios)

# Scenario 1: The Ideal Customer (High spend, frequent transactions)
evaluate_customer([9000, 85000, 45, 1000000, 0], "The Whale (VVIP)")

# Scenario 2: The "Ghosting" Customer (Spend dropped to near zero, low trans)
evaluate_customer([10, 500, 1, 500000, 0], "The Fading User (Silent Churn)")

# Scenario 3: The Average User
evaluate_customer([4000, 25000, 12, 150000, 0], "Regular Middle-Class User")

# Scenario 4: The Fraudulent/Problematic User
evaluate_customer([5000, 30000, 20, 500000, 1], "Customer with Fraud Flags")