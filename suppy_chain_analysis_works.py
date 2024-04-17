def load_and_clean_data(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    shipment_mode_mode = data['Shipment Mode'].mode()[0]
    data['Shipment Mode'] = data['Shipment Mode'].fillna(shipment_mode_mode)
    data['Freight Cost (USD)'] = pd.to_numeric(data['Freight Cost (USD)'], errors='coerce')
    freight_cost_median = data['Freight Cost (USD)'].median()
    data['Freight Cost (USD)'] = data['Freight Cost (USD)'].fillna(freight_cost_median)
    data['Weight (Kilograms)'] = pd.to_numeric(data['Weight (Kilograms)'], errors='coerce')
    data['Weight (Kilograms)'] = data['Weight (Kilograms)'].fillna(0)
    return data

def exploratory_data_analysis(data):
    print(data.info())
    print("\nStatistical Summary of Numerical Fields:")
    print(data.describe())
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    data['Line Item Quantity'].hist(bins=50)
    plt.title('Distribution of Line Item Quantity')
    plt.subplot(1, 2, 2)
    data['Line Item Value'].hist(bins=50)
    plt.title('Distribution of Line Item Value')
    plt.show()
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data['Freight Cost (USD)'])
    plt.title('Box Plot of Freight Cost (USD)')
    plt.show()
    sns.scatterplot(x='Line Item Value', y='Freight Cost (USD)', data=data)
    plt.title('Scatter Plot between Line Item Value and Freight Cost')
    plt.show()
    numerical_data = data.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap of Correlation Matrix')
    plt.show()

def main():
    file_path = 'SCMS_Delivery_History_Dataset_20150929.csv'
    data = load_and_clean_data(file_path)
    exploratory_data_analysis(data)
    
    # Split the data
    X = data[['Line Item Value', 'Weight (Kilograms)']]  # Assuming these are the features you want to use
    y = data['Freight Cost (USD)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Evaluation:")
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

if __name__ == "__main__":
    main()