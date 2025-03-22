import numpy as np
import pandas as pd
import os
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Load Training Data
DATA_PATH = r"C:\Users\Batia\Downloads\RetailRocket rec sys"
MODEL_PATH = os.path.join(DATA_PATH, "Models")
os.makedirs(MODEL_PATH, exist_ok=True)

train_df = pd.read_csv(os.path.join(DATA_PATH, "train_retailrocket.csv"))

# Define event weights
event_weights = {"view": 1, "add_to_cart": 2, "purchase": 5}

# Apply event weights
if "event" in train_df.columns:
    train_df["interaction_score"] = train_df["event"].map(event_weights)
else:
    train_df["interaction_score"] = 1  # Default binary interactions
train_df["interaction_score"] = train_df["interaction_score"].fillna(0)

# Encode users and items
user_encoder = {id_: i for i, id_ in enumerate(train_df["visitorid"].unique())}
item_encoder = {id_: i for i, id_ in enumerate(train_df["itemid"].unique())}

train_df["user_id"] = train_df["visitorid"].map(user_encoder)
train_df["item_id"] = train_df["itemid"].map(item_encoder)

# Save encoders
with open(os.path.join(MODEL_PATH, "user_encoder.pkl"), "wb") as f:
    pickle.dump(user_encoder, f)

with open(os.path.join(MODEL_PATH, "item_encoder.pkl"), "wb") as f:
    pickle.dump(item_encoder, f)

print(" Encoders saved successfully!")

# Filter users with at least 5 interactions
train_df = train_df.groupby("user_id").filter(lambda x: len(x) >= 5)

# Prepare Training Data for XGBoost
user_ids = train_df["user_id"].values
item_ids = train_df["item_id"].values
interaction_labels = (train_df["interaction_score"] > 0).astype(int)  # Convert to binary

train_data = pd.DataFrame({"user_id": user_ids, "item_id": item_ids, "interaction": interaction_labels})

# Create Negative Samples (Random Non-Interacted Items)
num_negative_samples = len(train_data)
negative_samples = []
user_item_pairs = set(zip(train_data["user_id"], train_data["item_id"]))

for _ in range(num_negative_samples):
    random_user = np.random.choice(train_data["user_id"].unique())
    random_item = np.random.choice(train_data["item_id"].unique())

    if (random_user, random_item) not in user_item_pairs:
        negative_samples.append((random_user, random_item, 0))

negative_df = pd.DataFrame(negative_samples, columns=["user_id", "item_id", "interaction"])

# Combine positive and negative samples
train_data = pd.concat([train_data, negative_df])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    train_data[["user_id", "item_id"]], train_data["interaction"], test_size=0.2, random_state=42
)

# Convert data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train XGBoost Model
xgb_model = xgb.train(
    {"objective": "binary:logistic", "max_depth": 10, "eta": 0.1, "eval_metric": "logloss"},
    dtrain,
    num_boost_round=50
)

# Save the trained model
xgb_model.save_model(os.path.join(MODEL_PATH, "xgboost_model.json"))
print(" XGBoost model saved successfully!")

# Predict and Evaluate
y_pred = (xgb_model.predict(dtest) > 0.5).astype(int)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f" XGBoost Results:")
print(f" Precision: {precision:.4f}")
print(f" Recall: {recall:.4f}")
print(f" F1 Score: {f1:.4f}")