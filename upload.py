import boto3
import pandas as pd
from tqdm import tqdm  # pip install tqdm

# 1. Load CSV
df = pd.read_csv("customer_profiles.csv")

# 2. Rename column
df = df.rename(columns={"user_id": "customer_id"})

# 3. Limit rows to 80,000
df = df.head(80000)

# 4. Connect to DynamoDB
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table("CustomerProfiles")

# 5. Faster dict conversion
records = df.to_dict(orient="records")

# 6. Insert rows with progress bar
with table.batch_writer() as batch:
    for row in tqdm(records, total=len(records)):
        item = {k: str(v) for k, v in row.items()}  # cast all values to string
        batch.put_item(Item=item)

print("✅ 80,000 rows ingested into DynamoDB")