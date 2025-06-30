import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    Executes the full machine learning pipeline:
    1. Data Ingestion & Preprocessing
    2. Feature Engineering
    3. Data Splitting & Scaling
    4. Baseline Model Training & Evaluation
    5. Deep Learning Model Training & Evaluation
    6. Model Saving
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

    # 4. Baseline Models
    print("\n--- Training & Evaluating Baseline Models ---")
    
    # Linear Regression
    lr_model = model_developer.get_linear_regression_model()
    lr_model.fit(X_train_scaled, y_train)
    lr_preds_log = lr_model.predict(X_test_scaled)
    
    y_test_original = np.expm1(y_test)
    lr_preds_original = np.expm1(lr_preds_log)
    
    lr_mae = mean_absolute_error(y_test_original, lr_preds_original)
    lr_rmse = np.sqrt(mean_squared_error(y_test_original, lr_preds_original))
    print(f"Linear Regression -> MAE: {lr_mae:.2f} Wh, RMSE: {lr_rmse:.2f} Wh")

    # Random Forest
    rf_model = model_developer.get_random_forest_model()
    rf_model.fit(X_train_scaled, y_train)
    rf_preds_log = rf_model.predict(X_test_scaled)
    rf_preds_original = np.expm1(rf_preds_log)

    rf_mae = mean_absolute_error(y_test_original, rf_preds_original)
    rf_rmse = np.sqrt(mean_squared_error(y_test_original, rf_preds_original))
    print(f"Random Forest -> MAE: {rf_mae:.2f} Wh, RMSE: {rf_rmse:.2f} Wh")

    # 5. Deep Learning Model
    print("\n--- Training & Evaluating Deep Learning Model (LSTM) ---")
    TIME_STEPS = 6
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, TIME_STEPS)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, TIME_STEPS)

    lstm_model = model_developer.get_lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    lstm_model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    lstm_preds_log = lstm_model.predict(X_test_seq)
    y_test_seq_original = np.expm1(y_test_seq)
    lstm_preds_original = np.expm1(lstm_preds_log)

    lstm_mae = mean_absolute_error(y_test_seq_original, lstm_preds_original)
    lstm_rmse = np.sqrt(mean_squared_error(y_test_seq_original, lstm_preds_original))
    print(f"LSTM Model -> MAE: {lstm_mae:.2f} Wh, RMSE: {lstm_rmse:.2f} Wh")

    # 6. Model Saving
    print("\n--- Saving Models ---")
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    lstm_model.save('models/lstm_model.h5')
    print("Models saved successfully to the 'models' directory.")
