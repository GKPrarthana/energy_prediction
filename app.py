from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

# --- Load Model and Scaler ---
# This part is similar to the prediction_pipeline.py
try:
    model_path = os.path.join('models', 'random_forest_model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

# This is the list of all 33 features the model was trained on
FEATURE_ORDER = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 
                 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9', 
                 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint', 
                 'hour', 'day_of_week', 'month', 'appliances_lag_1h', 'appliances_lag_2h', 
                 'T_out_rolling_mean_1h', 'appliances_rolling_mean_1h']

@app.route('/', methods=['GET', 'POST'])
def predict():
    # Default value for prediction
    prediction_text = ""
    
    # If the form is submitted
    if request.method == 'POST' and model is not None:
        try:
            # --- 1. Get data from the form ---
            # We get the most important features from the user
            t_out = float(request.form['T_out'])
            rh_out = float(request.form['RH_out'])
            lights = float(request.form['lights'])
            
            # --- 2. Create a full feature DataFrame ---
            # Create a dictionary with default values for all features
            # This is a simple way to handle the many features the model needs
            data = {feature: 1.0 for feature in FEATURE_ORDER} # Start with a default
            
            # Update with the user's input
            data['T_out'] = t_out
            data['RH_out'] = rh_out
            data['lights'] = lights
            
            # Create time-based features from the current time
            now = pd.Timestamp.now()
            data['hour'] = now.hour
            data['day_of_week'] = now.dayofweek
            data['month'] = now.month
            
            # Use defaults for complex features (lags, rolling means)
            data['appliances_lag_1h'] = 60
            data['appliances_lag_2h'] = 60
            data['T_out_rolling_mean_1h'] = t_out # Use current temp as approximation
            data['appliances_rolling_mean_1h'] = 60
            
            # Convert to DataFrame in the correct order
            input_df = pd.DataFrame([data])[FEATURE_ORDER]
            
            # --- 3. Scale and Predict ---
            input_scaled = scaler.transform(input_df)
            prediction_log = model.predict(input_scaled)[0]
            prediction_original = np.expm1(prediction_log)
            
            prediction_text = f"{prediction_original:.2f} Wh"

        except Exception as e:
            prediction_text = f"Error: {e}"

    # Render the HTML page, passing the prediction result to it
    return render_template('index.html', prediction_result=prediction_text)

if __name__ == "__main__":
    # Runs the Flask app
    app.run(debug=True)

