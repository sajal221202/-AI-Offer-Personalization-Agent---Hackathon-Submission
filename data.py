import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv('loyalty_data_80k.csv')  # Use your actual filename

# Convert ALL columns to float64
df_ml_ready = df.copy()

# Handle non-numeric columns first
for col in df_ml_ready.columns:
    if df_ml_ready[col].dtype == 'object':
        # Try to convert to numeric
        df_ml_ready[col] = pd.to_numeric(df_ml_ready[col], errors='coerce')
    
# Fill any NaN values with 0
df_ml_ready = df_ml_ready.fillna(0)

# Convert everything to float64
df_ml_ready = df_ml_ready.astype('float64')

print(f"✅ All columns converted to: {df_ml_ready.dtypes.iloc[0]}")
print(f"Shape: {df_ml_ready.shape}")

# Save ML-ready data
df_ml_ready.to_csv('loyalty_data_ml_ready.csv', index=False)
print("💾 Saved as 'loyalty_data_ml_ready.csv'")
