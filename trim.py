import pandas as pd

# Load and sample 80k random rows
df = pd.read_csv('cleaned_loyalty_data.csv')  # Replace with your actual filename
print(f"Original data shape: {df.shape}")

# Get random 80k rows (better for training)
df_80k = df.sample(n=80000, random_state=42)  # random_state for reproducible results
print(f"Sampled data shape: {df_80k.shape}")

# Save to new file
df_80k.to_csv('loyalty_data_80k_random.csv', index=False)
print("✅ Saved random 80k rows to 'loyalty_data_80k_random.csv'")
