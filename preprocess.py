import pandas as pd
import os

file_name = 'data.csv'

if os.path.exists(file_name):
    # 1. LOAD MORE DATA
    # Taking 500,000 rows to ensure we get a diverse set of transaction dates
    print("Loading data... this might take a few seconds...")
    df = pd.read_csv(file_name, nrows=500000) 
    print("✅ Transaction data loaded!")

    # 2. CONVERT DATES
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

    # 3. GROUP BY CUSTOMER
    print("Creating customer profiles...")
    customer_df = df.groupby('cc_num').agg({
        'amt': ['mean', 'sum', 'count'],           
        'trans_date_trans_time': 'max',            
        'city_pop': 'first',                       
        'is_fraud': 'sum'                          
    })

    # Flatten columns
    customer_df.columns = ['avg_spend', 'total_spend', 'trans_count', 'last_trans', 'city_pop', 'fraud_history']

    # 4. CALCULATE RECENCY (The "Silent" Factor)
    latest_date = customer_df['last_trans'].max()
    customer_df['days_since_last_trans'] = (latest_date - customer_df['last_trans']).dt.days
    
    # 5. ADJUST CHURN LOGIC (The Fix)
    # If a customer hasn't used their card in 5 days, we mark them as 'churned' 
    # This creates a 'balanced' dataset for your project to work with.
    customer_df['is_churned'] = (customer_df['days_since_last_trans'] > 5).astype(int)

    # 6. PRINT STATISTICS (Important for your report!)
    print("\n--- Churn Distribution ---")
    print(customer_df['is_churned'].value_counts())
    
    # 7. SAVE
    # We drop the date column before saving because XGBoost only wants numbers
    customer_df.drop(columns=['last_trans']).to_csv('processed_data.csv')
    print("\n✅ Success: 'processed_data.csv' created with a balanced dataset!")

else:
    print("❌ Error: 'data.csv' not found!")