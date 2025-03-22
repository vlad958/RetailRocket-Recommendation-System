import pandas as pd
import os

DATA_PATH = r"c:\Users\Batia\Downloads\RetailRocket rec sys"

# Load datasets
print(" Loading RetailRocket data...")
events = pd.read_csv(os.path.join(DATA_PATH, "events.csv"))
items = pd.read_csv(os.path.join(DATA_PATH, "merged_item_properties.csv"))

# Keep only required event types
events = events[events["event"].isin(["view", "addtocart", "transaction"])]

# Merge metadata (brand, category, etc.)
items = items.drop_duplicates(subset=["itemid"])
merged = events.merge(items, left_on="itemid", right_on="itemid", how="left")

# Save processed dataset
merged.to_csv(os.path.join(DATA_PATH, "processed_retailrocket.csv"), index=False)
print(" Preprocessing Complete! Data Saved.")
