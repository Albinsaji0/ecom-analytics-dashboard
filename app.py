from flask import Flask
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random

# Initialize Flask and Dash
server = Flask(__name__)
app = Dash(__name__, server=server)

# Global variables for models
arima_model, lstm_model = None, None
days, sales, future_days_arima, future_days_lstm = None, None, None, None
predicted_sales_arima, predicted_sales_lstm = None, None
churn_model, anomalies = None, None
kmeans_segments, customer_data = None, None

import pandas as pd

def load_real_data():
    # Load the dataset into a DataFrame
    ecom_data = pd.read_csv('ecommerce.csv') 
    return ecom_data


def generate_sales_from_real_data():
    ecom_data = load_real_data()
    days = pd.to_datetime(ecom_data['Order Date'])[:30]  # First 30 days
    sales = ecom_data['Sales'][:30]  # Use the 'Sales' column for forecasting
    return days, sales

# ARIMA for Sales Forecasting
def train_arima_model():
    days, sales = generate_sales_from_real_data()
    model = sm.tsa.ARIMA(sales, order=(5, 1, 0))  # ARIMA model with specified order
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=7)  # Predict next 7 days
    return model_fit, forecast

# LSTM for Sales Forecasting
def train_lstm_model():
    days, sales = generate_sales_from_real_data()
    sales = sales.values.reshape((sales.shape[0], 1, 1))  # Reshape for LSTM
    model = Sequential()
    model.add(tf.keras.Input(shape=(1, 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(days.reshape(-1, 1), sales, epochs=300, verbose=0)  # Train model
    return model

# Train ARIMA and LSTM models
def initialize_sales_models():
    global arima_model, lstm_model, days, sales, future_days_arima, predicted_sales_arima, future_days_lstm, predicted_sales_lstm
    days, sales = generate_sales_from_real_data()

    # ARIMA Model
    arima_model, predicted_sales_arima = train_arima_model()
    future_days_arima = np.arange(31, 38)

    # LSTM Model
    lstm_model = train_lstm_model()
    future_days_lstm = np.arange(31, 38).reshape(-1, 1)
    predicted_sales_lstm = lstm_model.predict(future_days_lstm).flatten()

initialize_sales_models()

# Ensure the future days and predictions are lists
future_days_arima = future_days_arima.tolist()
predicted_sales_arima = predicted_sales_arima.tolist()

future_days_lstm = future_days_lstm.flatten().tolist()
predicted_sales_lstm = predicted_sales_lstm.tolist()

def generate_customer_behavior_data_for_clustering():
    ecom_data = load_real_data()
    data = ecom_data[['Product Category', 'Sales']] 
    return data


def customer_segmentation():
    global customer_data, kmeans_segments
    customer_data = generate_customer_behavior_data_for_clustering()
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(customer_data)
    kmeans_segments = kmeans.labels_

customer_segmentation()

# 3. Real-Time Data Simulation with Random Sales Fluctuations
def generate_fluctuating_sales():
    return [random.randint(200, 1000) + random.randint(-50, 50) for _ in range(5)]

@app.callback(
    Output('real-time-sales-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_real_time_sales(n):
    sales_data = generate_fluctuating_sales()
    fig = px.bar(x=["Product A", "Product B", "Product C", "Product D", "Product E"], y=sales_data, title="Real-Time Sales")
    return fig

# 4. Sales Anomaly Detection
def train_anomaly_model():
    global anomalies
    days, sales = generate_sales_from_real_data()
    model = IsolationForest(contamination=0.1)
    model.fit(sales.values.reshape(-1, 1))
    anomalies = model.predict(sales.values.reshape(-1, 1))
    return days, sales, anomalies

days, sales, anomalies = train_anomaly_model()

# Layout for the Sales Dashboard
sales_dashboard_layout = html.Div([
    html.H1("Sales Dashboard"),
    dcc.Graph(id='actual-sales-graph', figure=px.line(x=days, y=sales, title="Actual Sales (Last 30 Days)", labels={"x": "Days", "y": "Sales"}, template="plotly_dark")),
    dcc.Graph(id='predicted-sales-arima-graph', figure=px.line(x=future_days_arima, y=predicted_sales_arima, title="ARIMA Predicted Sales (Next 7 Days)", labels={"x": "Future Days", "y": "Predicted Sales"}, template="plotly_dark")),
    dcc.Graph(id='predicted-sales-lstm-graph', figure=px.line(x=future_days_lstm, y=predicted_sales_lstm, title="LSTM Predicted Sales (Next 7 Days)", labels={"x": "Future Days", "y": "Predicted Sales"}, template="plotly_dark")),
    dcc.Graph(id='real-time-sales-graph'),  # Real-time graph
    dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0),
    dcc.Link('Go to AI Insights', href='/ai-insights')
])

# AI Insights Layout
ai_insights_layout = html.Div([
    html.H1("AI Insights"),
    dcc.Graph(id='customer-segmentation-graph'),
    dcc.Graph(id='churn-prediction-graph'),
    dcc.Graph(id='anomaly-detection-graph'),
    dcc.Link('Go to Sales Dashboard', href='/')
])

# Main layout with navigation
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Page navigation
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/ai-insights':
        return ai_insights_layout
    else:
        return sales_dashboard_layout

# Customer Segmentation Graph
@app.callback(
    Output('customer-segmentation-graph', 'figure'),
    [Input('url', 'pathname')]
)
def update_customer_segmentation(pathname):
    if pathname == '/ai-insights':
        fig = px.scatter(x=customer_data['total_spend'], y=customer_data['purchase_count'], color=kmeans_segments, title="Customer Segmentation (KMeans Clustering)", labels={"x": "Total Spend", "y": "Purchase Count"})
        return fig
    return {}

# Churn Prediction
@app.callback(
    Output('churn-prediction-graph', 'figure'),
    [Input('url', 'pathname')]
)
def predict_churn(pathname):
    if pathname == '/ai-insights':
        data = generate_customer_behavior_data_for_clustering()
        churn_pred = churn_model.predict(data)
        fig = px.bar(x=data.index, y=churn_pred, title="Churn Predictions")
        return fig
    return {}

# Sales Anomaly Detection Graph
@app.callback(
    Output('anomaly-detection-graph', 'figure'),
    [Input('url', 'pathname')]
)
def update_anomaly_graph(pathname):
    if pathname == '/ai-insights':
        anomaly_sales = np.where(anomalies == -1, sales, np.nan)
        fig = px.scatter(x=days, y=sales, title="Sales Anomaly Detection")
        fig.add_scatter(x=days, y=anomaly_sales, mode='markers', marker=dict(color='red', size=12), name='Anomaly')
        return fig
    return {}

# Run Flask server
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
