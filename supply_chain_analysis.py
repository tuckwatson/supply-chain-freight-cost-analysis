import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_data(file_path):
    """
    Used to load data from CSV and perform initial cleaning operations.
    """
    logging.info("Loading data from file: %s", file_path)
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    # Handle missing values and convert types
    data['Shipment Mode'] = data['Shipment Mode'].fillna(data['Shipment Mode'].mode()[0])
    data['Freight Cost (USD)'] = pd.to_numeric(data['Freight Cost (USD)'], errors='coerce')
    data['Freight Cost (USD)'] = data['Freight Cost (USD)'].fillna(data['Freight Cost (USD)'].median())
    return data

def perform_eda(data):
    """
    Perform exploratory data analysis and generate visualizations.
    """
    logging.info("Performing exploratory data analysis.")
    plt.figure(figsize=(12, 6))
    sns.histplot(data['Line Item Quantity'], bins=50, kde=True)
    plt.title('Distribution of Line Item Quantity')
    plt.show()

def train_model(X, y):
    """
    Function used to train our linear regression model.
    """
    logging.info("Training model.")
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5)
    model.fit(X, y)
    logging.info("Model training completed with average cross-validation score: %f", np.mean(scores))
    return model

def main():
    file_path = 'SCMS_Delivery_History_Dataset_20150929.csv'
    data = load_and_clean_data(file_path)
    perform_eda(data)
    X = data[['Line Item Value', 'Weight (Kilograms)']]
    y = data['Freight Cost (USD)']
    model = train_model(X, y)
    # Save the trained model
    import joblib
    joblib.dump(model, 'model.pkl')
    logging.info("Model saved as model.pkl")

if __name__ == "__main__":
    main()