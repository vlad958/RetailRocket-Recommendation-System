import numpy as np
import pandas as pd
import os
import pickle
import xgboost as xgb
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Load Model and Encoders
DATA_PATH = r"C:\Users\Batia\Downloads\RetailRocket rec sys"
MODEL_PATH = os.path.join(DATA_PATH, "Models")

xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(MODEL_PATH, "xgboost_model.json"))

with open(os.path.join(MODEL_PATH, "user_encoder.pkl"), "rb") as f:
    user_encoder = pickle.load(f)

with open(os.path.join(MODEL_PATH, "item_encoder.pkl"), "rb") as f:
    item_encoder = pickle.load(f)

# Load Dataset for EDA
df = pd.read_csv(os.path.join(DATA_PATH, "train_retailrocket.csv"))
if df["timestamp"].dtype != 'datetime64[ns]':
    df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date

# Dash App Initialization
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "RetailRocket Recommendation System"

# Layout with Tabs
app.layout = html.Div([
    dcc.Tabs(id="tabs", value="recommend", children=[
        dcc.Tab(label="Recommendations", value="recommend"),
        dcc.Tab(label="Data Preview", value="data_preview"),
        dcc.Tab(label="Exploratory Data Analysis", value="eda")
    ]),
    html.Div(id="tabs-content")
])

# Recommendation Function
def recommend_products(user_id, model, item_encoder, top_n=10):
    item_ids = list(item_encoder.values())
    user_input_df = pd.DataFrame([[user_id, item] for item in item_ids], columns=["user_id", "item_id"])
    dmatrix = xgb.DMatrix(user_input_df, feature_names=["user_id", "item_id"])
    scores = model.predict(dmatrix)
    top_items = np.argsort(scores)[-top_n:][::-1]
    recommended_items = [list(item_encoder.keys())[list(item_encoder.values()).index(item)] for item in top_items]
    return recommended_items

# Generate Figures
def generate_pie_chart():
    fig = px.pie(names=df["event"].value_counts().index, values=df["event"].value_counts().values, 
                 title="Event Type Distribution", hole=0.3)
    return fig

def generate_bar_chart(column, title):
    top_values = df[column].value_counts().head(10)
    fig = px.bar(x=top_values.index, y=top_values.values, labels={"x": column, "y": "Count"}, title=title)
    return fig

def generate_line_chart(filtered_events):
    event_time_series_filtered = filtered_events.groupby(["date", "event"]).size().unstack().fillna(0)
    fig = px.line(event_time_series_filtered, title="Add to Cart & Purchase Events Over Time")
    return fig

# Callback to Update Tabs
@app.callback(Output("tabs-content", "children"), [Input("tabs", "value")])
def update_tab(tab_name):
    if tab_name == "recommend":
        return html.Div([
            html.H1("RetailRocket Recommendation System"),
            html.Label("Select User ID:"),
            dcc.Dropdown(
                id="user_id_dropdown",
                options=[{"label": str(user_id), "value": user_id} for user_id in user_encoder.values()],
                value=list(user_encoder.values())[0]
            ),
            html.Button("Get Recommendations", id="recommend_btn", n_clicks=0),
            html.H3("Recommended Items:"),
            dash_table.DataTable(id="recommendations_table")
        ])
    elif tab_name == "data_preview":
        sample_data = pd.DataFrame({
            "User ID": list(user_encoder.keys())[:10],
            "Encoded User ID": list(user_encoder.values())[:10],
            "Item ID": list(item_encoder.keys())[:10],
            "Encoded Item ID": list(item_encoder.values())[:10]
        })
        return html.Div([
            html.H1("Data Preview"),
            dash_table.DataTable(
                id="data_table",
                columns=[{"name": col, "id": col} for col in sample_data.columns],
                data=sample_data.to_dict("records"),
                page_size=10
            )
        ])
    elif tab_name == "eda":
        filtered_events = df[df["event"].isin(["addtocart", "transaction"])]
        return html.Div([
            html.H1("Exploratory Data Analysis"),
            dcc.Graph(figure=generate_pie_chart()),
            dcc.Graph(figure=generate_bar_chart("visitorid", "Top 10 Most Active Users")),
            dcc.Graph(figure=generate_bar_chart("itemid", "Top 10 Most Popular Items")),
            dcc.Graph(figure=generate_bar_chart("itemid", "Top 10 Most Purchased Items")),
            dcc.Graph(figure=generate_line_chart(filtered_events))
        ])
    return html.Div()

# Callback for Recommendations
@app.callback(
    Output("recommendations_table", "data"),
    Input("recommend_btn", "n_clicks"),
    Input("user_id_dropdown", "value")
)
def update_recommendations(n_clicks, user_id):
    if n_clicks > 0:
        recommended_items = recommend_products(user_id, xgb_model, item_encoder)
        return [{"Recommended Items": item} for item in recommended_items]
    return []

if __name__ == "__main__":
    app.run_server(debug=True)
