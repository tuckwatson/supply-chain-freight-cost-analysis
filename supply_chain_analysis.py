import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Adding StandardScaler and OneHotEncoder

def load_and_clean_data(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    data['Shipment Mode'] = data['Shipment Mode'].fillna(data['Shipment Mode'].mode()[0])
    data['Freight Cost (USD)'] = pd.to_numeric(data['Freight Cost (USD)'], errors='coerce')
    data['Freight Cost (USD)'] = data['Freight Cost (USD)'].fillna(data['Freight Cost (USD)'].median())
    data['Line Item Value'] = pd.to_numeric(data['Line Item Value'], errors='coerce').fillna(0)
    return data

def main():
    file_path = 'SCMS_Delivery_History_Dataset_20150929.csv'
    data = load_and_clean_data(file_path)
    
    # One-Hot Encoding
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(data[['Shipment Mode']]).toarray()  # Use toarray() to convert to dense format
    
    # Feature Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[['Line Item Value', 'Freight Cost (USD)']])

    X = data[['Line Item Value', 'Freight Cost (USD)']].fillna(0)
    y = data['Freight Cost (USD)'].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training a RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Random Forest Model Evaluation:")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data['Line Item Value'], y=data['Freight Cost (USD)'])
    plt.title('Scatter Plot of Line Item Value vs Freight Cost')
    plt.show()

if __name__ == "__main__":
    main()
