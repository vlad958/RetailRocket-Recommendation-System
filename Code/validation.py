import numpy as np
import pandas as pd
import os
import pickle
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

# Load Validation Data
DATA_PATH = r"C:\Users\Batia\Downloads\RetailRocket rec sys"
MODEL_PATH = os.path.join(DATA_PATH, "Models")
val_df = pd.read_csv(os.path.join(DATA_PATH, "validation_retailrocket.csv"))

# Load trained XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(MODEL_PATH, "xgboost_model.json"))
print(" XGBoost model loaded successfully!")

# Load Encoders
with open(os.path.join(MODEL_PATH, "user_encoder.pkl"), "rb") as f:
    user_encoder = pickle.load(f)

with open(os.path.join(MODEL_PATH, "item_encoder.pkl"), "rb") as f:
    item_encoder = pickle.load(f)

print(" Encoders loaded successfully!")

# Apply event weights
event_weights = {"view": 1, "add_to_cart": 2, "purchase": 5}
if "event" in val_df.columns:
    val_df["interaction_score"] = val_df["event"].map(event_weights)
else:
    val_df["interaction_score"] = 1
val_df["interaction_score"] = val_df["interaction_score"].fillna(0)

# Encode users and items
val_df = val_df[val_df["visitorid"].isin(user_encoder)]
val_df = val_df[val_df["itemid"].isin(item_encoder)]
val_df["user_id"] = val_df["visitorid"].map(user_encoder)
val_df["item_id"] = val_df["itemid"].map(item_encoder)

# Prepare Validation Data
X_val = val_df[["user_id", "item_id"]]
y_val = (val_df["interaction_score"] > 0).astype(int)

dval = xgb.DMatrix(X_val)

# Predict and Evaluate
y_pred = (xgb_model.predict(dval) > 0.5).astype(int)
precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)

print(f" XGBoost Validation Results:")
print(f" Precision: {precision:.4f}")
print(f" Recall: {recall:.4f}")
print(f" F1 Score: {f1:.4f}")
