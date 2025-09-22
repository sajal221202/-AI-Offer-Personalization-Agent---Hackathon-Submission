# Complete Loyalty Data Cleaning Script for Cursor
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def clean_loyalty_data(csv_file_path):
    """Clean your loyalty program data"""
    print("🚀 Starting Loyalty Data Cleaning...")
    
    # Load data
    df = pd.read_csv(csv_file_path)
    print(f"✅ Data loaded: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display basic info
    print(f"\n📊 Data Overview:")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Clean user_id (essential for loyalty system)
    if 'user_id' in df.columns:
        initial_count = len(df)
        df = df.dropna(subset=['user_id'])
        removed = initial_count - len(df)
        if removed > 0:
            print(f"✅ Removed {removed} records with missing user_id")
    
    # Clean customer names
    if 'first_name' in df.columns:
        df['first_name'] = df['first_name'].fillna('Unknown')
        print("✅ Cleaned customer names")
    
    # Clean age data
    if 'age' in df.columns:
        # Fix unrealistic ages
        df.loc[(df['age'] < 13) | (df['age'] > 100), 'age'] = np.nan
        df['age'].fillna(df['age'].median(), inplace=True)
        
        # Create age segments for targeting
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 25, 35, 50, 65, 100], 
                                labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        print("✅ Cleaned age and created age groups")
    
    # Clean loyalty tiers
    if 'loyalty_tier' in df.columns:
        df['loyalty_tier'] = df['loyalty_tier'].fillna('Bronze')
        
        # Standardize tier names
        tier_mapping = {
            'bronze': 'Bronze', 'silver': 'Silver', 'gold': 'Gold', 'platinum': 'Platinum',
            'BRONZE': 'Bronze', 'SILVER': 'Silver', 'GOLD': 'Gold', 'PLATINUM': 'Platinum'
        }
        df['loyalty_tier'] = df['loyalty_tier'].replace(tier_mapping)
        
        # Create numeric tier levels
        tier_levels = {'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4}
        df['tier_level'] = df['loyalty_tier'].map(tier_levels).fillna(1)
        print("✅ Standardized loyalty tiers")
    
    # Clean transaction amounts
    if 'bill_amount' in df.columns:
        # Remove currency symbols if data is stored as text
        if df['bill_amount'].dtype == 'object':
            df['bill_amount'] = df['bill_amount'].str.replace(r'[₹$£€¥,]', '', regex=True)
            df['bill_amount'] = pd.to_numeric(df['bill_amount'], errors='coerce')
        
        # Fill missing amounts with median
        df['bill_amount'].fillna(df['bill_amount'].median(), inplace=True)
        
        # Create transaction size categories
        df['transaction_size'] = pd.cut(df['bill_amount'],
                                      bins=[0, 500, 2000, 10000, float('inf')],
                                      labels=['Small', 'Medium', 'Large', 'VIP'])
        
        # Flag high-value transactions
        df['high_value_transaction'] = (df['bill_amount'] > 10000).astype(int)
        print("✅ Cleaned transaction amounts")
    
    # Clean points data
    for points_col in ['points_earned', 'points_redeemed']:
        if points_col in df.columns:
            df[points_col].fillna(0, inplace=True)
            # Remove negative points (data errors)
            df.loc[df[points_col] < 0, points_col] = 0
            print(f"✅ Cleaned {points_col}")
    
    # Clean coupon data
    for coupon_col in ['total_coupons_issued', 'coupons_redeemed_in_bill']:
        if coupon_col in df.columns:
            df[coupon_col].fillna(0, inplace=True)
            df[coupon_col] = df[coupon_col].astype(int)
            print(f"✅ Cleaned {coupon_col}")
    
    # Clean dates
    date_columns = ['transaction_date', 'last_transaction_date', 'date_of_birth']
    for date_col in date_columns:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            print(f"✅ Cleaned {date_col}")
    
    # Create time-based features
    if 'transaction_date' in df.columns:
        df['transaction_year'] = df['transaction_date'].dt.year
        df['transaction_month'] = df['transaction_date'].dt.month
        df['transaction_day_of_week'] = df['transaction_date'].dt.dayofweek
        df['weekend'] = (df['transaction_day_of_week'] >= 5).astype(int)
        
        # Time of day categories
        df['transaction_hour'] = df['transaction_date'].dt.hour
        df['time_of_day'] = pd.cut(df['transaction_hour'],
                                 bins=[-1, 6, 12, 18, 24],
                                 labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        print("✅ Created time-based features")
    
    # Clean location data
    if 'store_name' in df.columns:
        df['store_name'] = df['store_name'].fillna('Unknown Store')
        # Create store popularity metric
        store_counts = df['store_name'].value_counts()
        df['store_popularity'] = df['store_name'].map(store_counts)
        print("✅ Cleaned store data")
    
    if 'zone' in df.columns:
        df['zone'] = df['zone'].fillna('Unknown')
        print("✅ Cleaned zone data")
    
    # Create loyalty engagement metrics
    if all(col in df.columns for col in ['points_earned', 'points_redeemed']):
        df['points_utilization_rate'] = np.where(
            df['points_earned'] > 0,
            df['points_redeemed'] / df['points_earned'],
            0
        )
        df['net_points_balance'] = df['points_earned'] - df['points_redeemed']
        print("✅ Created points engagement metrics")
    
    if all(col in df.columns for col in ['total_coupons_issued', 'coupons_redeemed_in_bill']):
        df['coupon_utilization_rate'] = np.where(
            df['total_coupons_issued'] > 0,
            df['coupons_redeemed_in_bill'] / df['total_coupons_issued'],
            0
        )
        print("✅ Created coupon engagement metrics")
    
    # Calculate recency if both dates available
    if all(col in df.columns for col in ['transaction_date', 'last_transaction_date']):
        df['days_since_last_transaction'] = (
            df['transaction_date'] - df['last_transaction_date']
        ).dt.days
        # Fix negative values (data quality issues)
        df.loc[df['days_since_last_transaction'] < 0, 'days_since_last_transaction'] = 0
        print("✅ Calculated customer recency")
    
    print(f"\n🎉 Data cleaning completed!")
    print(f"Final dataset shape: {df.shape}")
    print(f"Missing values remaining: {df.isnull().sum().sum()}")
    
    return df

def create_customer_summary(df):
    """Create customer-level summary for ML models"""
    if 'user_id' not in df.columns:
        print("❌ Cannot create customer summary - user_id missing")
        return None
    
    print("\n👥 Creating customer summary...")
    
    # Basic aggregations
    agg_dict = {}
    if 'bill_amount' in df.columns:
        agg_dict['bill_amount'] = ['count', 'sum', 'mean', 'std']
    if 'points_earned' in df.columns:
        agg_dict['points_earned'] = ['sum', 'mean']
    if 'points_redeemed' in df.columns:
        agg_dict['points_redeemed'] = ['sum', 'mean']
    
    customer_summary = df.groupby('user_id').agg(agg_dict).round(2)
    
    # Flatten column names
    customer_summary.columns = ['_'.join(col).strip() for col in customer_summary.columns.values]
    
    # Add custom metrics
    if 'bill_amount_count' in customer_summary.columns:
        customer_summary['transaction_frequency'] = customer_summary['bill_amount_count']
        customer_summary['total_spend'] = customer_summary['bill_amount_sum']
        customer_summary['avg_transaction_value'] = customer_summary['bill_amount_mean']
    
    # Add latest tier and demographics
    latest_data = df.groupby('user_id').last()[['loyalty_tier', 'age', 'zone']].fillna('Unknown')
    customer_summary = customer_summary.join(latest_data)
    
    print(f"✅ Customer summary created: {customer_summary.shape}")
    return customer_summary

# Usage function
def process_loyalty_data(csv_file_path, save_results=True):
    """Complete data processing pipeline"""
    print("="*60)
    print("🎯 LOYALTY DATA PROCESSING PIPELINE")
    print("="*60)
    
    # Clean the data
    df_clean = clean_loyalty_data(csv_file_path)
    
    # Create customer summary
    customer_summary = create_customer_summary(df_clean)
    
    # Save results
    if save_results:
        df_clean.to_csv('cleaned_loyalty_data.csv', index=False)
        if customer_summary is not None:
            customer_summary.to_csv('customer_summary.csv', index=True)
        print(f"\n💾 Files saved:")
        print(f"  - cleaned_loyalty_data.csv ({df_clean.shape})")
        print(f"  - customer_summary.csv ({customer_summary.shape if customer_summary is not None else 'N/A'})")
    
    # Display final summary
    print(f"\n📊 FINAL SUMMARY:")
    print(f"  Total customers: {df_clean['user_id'].nunique() if 'user_id' in df_clean.columns else 'N/A'}")
    print(f"  Total transactions: {len(df_clean)}")
    if 'bill_amount' in df_clean.columns:
        print(f"  Total revenue: ₹{df_clean['bill_amount'].sum():,.2f}")
        print(f"  Average transaction: ₹{df_clean['bill_amount'].mean():.2f}")
    if 'loyalty_tier' in df_clean.columns:
        print(f"  Loyalty tier distribution:")
        tier_dist = df_clean['loyalty_tier'].value_counts()
        for tier, count in tier_dist.items():
            print(f"    {tier}: {count} customers")
    
    print(f"\n🚀 Data ready for Multi-Agent AI System!")
    
    return df_clean, customer_summary

# Example usage:
if __name__ == "__main__":
    # Replace 'your_file.csv' with your actual CSV file path
    # df_cleaned, customer_features = process_loyalty_data('your_file.csv')
    print("Script ready! Use: process_loyalty_data('transaction_level.csv')")
