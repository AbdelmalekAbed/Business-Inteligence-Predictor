import pandas as pd
df = pd.read_csv("data/retail_sales.csv", parse_dates=["date"])
print("Rows, cols:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nHead:\n", df.head())
print("\nDescribe numeric:\n", df.describe())
print("\nValue counts (region):\n", df["region"].value_counts())
print("\nValue counts (product_category):\n", df["product_category"].value_counts())
