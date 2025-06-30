import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

# Import components from the project structure
from appliance_energy_predictor.components.data_preprocessing import DataPreprocessing
from appliance_energy_predictor.components.feature_engineering import FeatureEngineering
from appliance_energy_predictor.components.model_dev import ModelDevelopment

def create_sequences(X, y, time_steps=1):
    """Helper function to create 3D sequences for LSTM models."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def run_pipeline():
    """
    Executes the full machine learning pipeline.
    """
    # 1. Data Ingestion & Preprocessing
    data_preprocessor = DataPreprocessing()
    df_cleaned = data_preprocessor.load_and_clean_data()
    if df_cleaned is None:
        return

    # 2. Feature Engineering
    feature_engineer = FeatureEngineering(df_cleaned)
    df_featured = feature_engineer.create_features()

    # 3. Data Splitting & Scaling
    features = [col for col in df_featured.columns if col != 'Appliances']
    target = 'Appliances'
    X = df_featured[features]
    y = df_featured[target]

    split_index = int(len(df_featured) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data splitting and scaling complete.")

    # Instantiate model developer
    model_developer = ModelDevelopment()

    # 4. Train Best Model (Random Forest)
    print("\n--- Training Best Model (Random Forest) ---")
    rf_model = model_developer.get_random_forest_model()
    rf_model.fit(X_train_scaled, y_train)
    
    # 5. Model Saving
    print("\n--- Saving Model and Scaler ---")
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl') # <--- THIS LINE SAVES THE SCALER
    print("Model and scaler saved successfully to the 'models' directory.")

