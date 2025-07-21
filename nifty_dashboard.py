import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

# Download Nifty 50 Data
def get_nifty_50_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Calculate Log Returns
def calculate_log_returns(data):
    log_returns = np.log(data / data.shift(1)).dropna()
    return log_returns

# Plot Time Series
def plot_time_series(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    data.plot(ax=ax)
    ax.set_title("Nifty 50 Stock Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)")
    ax.legend(loc='upper left')
    st.pyplot(fig)

# Plot Log Returns
def plot_log_returns(log_returns):
    fig, ax = plt.subplots(figsize=(12, 6))
    log_returns.plot(ax=ax)
    ax.set_title("Daily Log Returns - Nifty 50 Stocks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    ax.legend(loc='upper left')
    st.pyplot(fig)

# Plot Correlation Heatmap
def plot_correlation_heatmap(log_returns):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(log_returns.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix - Nifty 50 Stocks")
    st.pyplot(fig)

# Plot Distance Heatmap
def plot_distance_heatmap(log_returns):
    distances = pdist(log_returns.T, metric='euclidean')
    distance_matrix = pd.DataFrame(squareform(distances), index=log_returns.columns, columns=log_returns.columns)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(distance_matrix, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Distance Matrix - Nifty 50 Stocks")
    st.pyplot(fig)

# Plot MDS
def plot_mds(log_returns):
    distances = pdist(log_returns.T, metric='euclidean')
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    mds_results = mds.fit_transform(squareform(distances))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(mds_results[:, 0], mds_results[:, 1], color='blue')
    for i, ticker in enumerate(log_returns.columns):
        ax.text(mds_results[i, 0], mds_results[i, 1], ticker)
    ax.set_title("MDS Plot - Nifty 50 Stocks")
    ax.set_xlabel("MDS Dimension 1")
    ax.set_ylabel("MDS Dimension 2")
    st.pyplot(fig)

# Plot KMeans Elbow Method
def plot_kmeans_elbow(log_returns):
    inertia = []
    X = log_returns.T  # Transpose to have stocks as data points
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, 11), inertia, marker='o')
    ax.set_title("Elbow Method for Optimal K")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

# Plot KMeans Clusters
def plot_kmeans_clusters(log_returns, optimal_clusters=3):
    X = log_returns.T
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, label in enumerate(X.index):
        ax.scatter(X.iloc[i, 0], X.iloc[i, 1], label=label, c=f"C{clusters[i]}")
    
    ax.set_title(f"KMeans Clustering - {optimal_clusters} Clusters")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

# Streamlit App Layout
st.title("Nifty 50 Stock Analysis Dashboard")

# Sidebar options
tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
    "ITC.NS", "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "HCLTECH.NS",
    "MARUTI.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS",
    "SUNPHARMA.NS", "NESTLEIND.NS", "POWERGRID.NS", "BAJAJFINSV.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "M&M.NS", "INDUSINDBK.NS", "DIVISLAB.NS", "TECHM.NS",
    "DRREDDY.NS", "GRASIM.NS", "HINDALCO.NS", "ADANIPORTS.NS", "SBILIFE.NS",
    "HEROMOTOCO.NS", "EICHERMOT.NS", "BRITANNIA.NS", "BPCL.NS", "COALINDIA.NS",
    "UPL.NS", "SHREECEM.NS", "ONGC.NS", "CIPLA.NS", "APOLLOHOSP.NS",
    "TATAMOTORS.NS", "VEDL.NS", "GAIL.NS", "IOC.NS", "BHARTIARTL.NS"
]

selected_tickers = st.sidebar.multiselect("Select Stocks", tickers, default=tickers[:5])
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# Load and display data
if selected_tickers:
    data = get_nifty_50_data(selected_tickers, start_date, end_date)
    log_returns = calculate_log_returns(data)

    st.subheader("Time Series Plot")
    plot_time_series(data)
    
    st.subheader("Log Returns Plot")
    plot_log_returns(log_returns)
    
    st.subheader("Correlation Matrix Heatmap")
    plot_correlation_heatmap(log_returns)
    
    st.subheader("Distance Matrix Heatmap")
    plot_distance_heatmap(log_returns)
    
    st.subheader("Multidimensional Scaling (MDS) Plot")
    plot_mds(log_returns)
    
    st.subheader("Elbow Method for Optimal Clusters")
    plot_kmeans_elbow(log_returns)
    
    optimal_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)
    st.subheader(f"KMeans Clustering with {optimal_clusters} Clusters")
    plot_kmeans_clusters(log_returns, optimal_clusters)
