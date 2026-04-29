import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os

def detect_drift(reference, current, column):
    # Performing the Kolmogorov-Smirnov test (Industry standard for drift)
    statistic, p_value = ks_2samp(reference[column], current[column])
    return p_value

# 1. Load Data
if os.path.exists('processed_data.csv'):
    df = pd.read_csv('processed_data.csv')
    
    # Create Reference (Training data) and Current (New data)
    reference = df.head(len(df)//2)
    current = df.tail(len(df)//2).copy()
    
    # 2. Simulate Drift (Economic Shift)
    # We drop the spending by 50% in the 'current' group
    current['avg_spend'] = current['avg_spend'] * 0.5
    
    print("--- FinTech Model Monitoring Report ---")
    print(f"Analyzing {len(df)} customer profiles...\n")
    
    # 3. Analyze key features
    features = ['avg_spend', 'total_spend', 'city_pop']
    
    for feature in features:
        p_val = detect_drift(reference, current, feature)
        status = "✅ STABLE" if p_val > 0.05 else "🚨 DRIFT DETECTED"
        print(f"Feature: {feature.ljust(12)} | P-Value: {p_val:.4f} | Status: {status}")

    print("\n--- Summary ---")
    print("Action Required: Retrain model on new spending patterns.")
    
else:
    print("❌ Error: processed_data.csv not found.")