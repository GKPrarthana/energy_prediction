from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

class ModelDevelopment:
    def get_linear_regression_model(self):
        """Returns a Linear Regression model instance."""
        return LinearRegression()

    def get_random_forest_model(self):
        """Returns a RandomForestRegressor model instance with predefined settings."""
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    def get_lstm_model(self, input_shape):
        """
        Builds and returns an LSTM model with a specified input shape.
        Args:
            input_shape (tuple): The shape of the input data (time_steps, n_features).
        
        Returns:
            tensorflow.keras.Model: The compiled LSTM model.
        """
        model = Sequential()
        model.add(LSTM(
            units=50,
            activation='relu',
            input_shape=input_shape
        ))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        print("LSTM model built successfully.")
        model.summary()
        return model

