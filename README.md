# 🎮 Game Sales Prediction App
## 📘 Overview

The Game Sales Prediction App is a machine learning web application built using Streamlit that predicts Global Game Sales (in millions) based on various regional sales, game attributes (platform, genre, publisher), and year of release.
It connects to a PostgreSQL database, processes data using Pandas and Scikit-Learn, and visualizes it interactively with Plotly.

## 🚀 Features

🔮 Predicts Global Game Sales using an ensemble of regression models (Voting Regressor).

📊 Displays model performance metrics (R², MAE, MSE).

🌍 Interactive data exploration with Plotly scatter plots.

💾 Fetches real data directly from a PostgreSQL database.

🧠 Combines multiple models for better accuracy.

## 🧩 Tech Stack
Component	Technology Used
Frontend	Streamlit
Backend	Python
Database	PostgreSQL
Machine Learning	scikit-learn
Data Handling	pandas
Visualization	Plotly
Deployment	Streamlit Cloud / Localhost

## 🧠 Model Details

The app uses an ensemble learning approach called Voting Regressor, which combines predictions from multiple models to improve performance.

Models Used:

KNeighborsRegressor → Captures local data patterns

DecisionTreeRegressor → Handles non-linear relationships

LinearRegression → Learns general trends

Preprocessing:

Ordinal Encoding for categorical variables (platform, genre, publisher)

Train-Test Split: 80% training, 20% testing

Evaluation Metrics:

R² Score

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

## 📊 Example Output

Input game details (year, genre, etc.)

Click “Predict Global Sales”

Output example:

🌍 Predicted Global Sales: 4.32 million units

You’ll also see:

Model metrics (R², MAE, MSE)

Interactive scatter plot showing game sales by year & genre
