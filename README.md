# Nifty_dashboard_Analysis
📊 Nifty 50 Dashboard Analysis
An interactive Streamlit-based dashboard for exploring Nifty 50 stock trends, returns, correlations, clustering, and market behavior through visual analytics and machine learning techniques.


🔍 Features
📈 Stock Price Time Series: Visualize historical adjusted closing prices of selected Nifty 50 stocks.

🔁 Log Returns: Calculate and plot daily log returns for return behavior analysis.

🔥 Correlation Matrix: See how Nifty 50 stocks move together.

📏 Distance Matrix: Understand pairwise dissimilarity using Euclidean distance.

🌐 MDS Plot: View 2D representation of stocks based on distances using Multidimensional Scaling.

🤖 KMeans Clustering:

Elbow method to find optimal number of clusters.

Cluster visualization with interactive slider.


🧠 Technologies Used
Streamlit – UI and interactivity

yfinance – Stock data fetching

pandas & numpy – Data manipulation

matplotlib & seaborn – Visualization

scikit-learn – MDS and KMeans clustering

scipy – Distance matrix computation


📅 Default Settings
Start Date: 2023-01-01

End Date: 2023-12-31

Default Tickers: Top 5 Nifty 50 companies


🚀 How to Run
git clone https://github.com/ananya1510/nifty_dashboard.git
cd nifty-50-dashboard
pip install -r requirements.txt
streamlit run app.py


🙌 Contributions Welcome!
If you have ideas to improve the dashboard (e.g., sector filters, performance stats, or predictive models), feel free to fork, enhance, and open a pull request.

📝 License
This project is licensed under the MIT License.




