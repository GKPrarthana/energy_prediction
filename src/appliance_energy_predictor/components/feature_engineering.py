import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self, df):
        """
        Initializes the feature engineering component.
        Args:
            df (pd.DataFrame): The preprocessed DataFrame.
        """
        self.df = df

    def create_features(self):
        """
        Engineers new features to improve model performance.
        - Creates time-based features (hour, day of week, month).
        - Creates lagged features for the target variable.
        - Creates rolling window features for key variables.
        - Drops NaN values resulting from these operations.
        
        Returns:
            pd.DataFrame: The DataFrame with engineered features.
        """
        df_featured = self.df.copy()
        
        # Create time-based features
        df_featured['hour'] = df_featured.index.hour
        df_featured['day_of_week'] = df_featured.index.dayofweek
        df_featured['month'] = df_featured.index.month
        
        # Create lagged features
        # We use the original (non-transformed) 'Appliances' for lags for clarity
        original_appliances = np.expm1(df_featured['Appliances'])
        df_featured['appliances_lag_1h'] = original_appliances.shift(6)
        df_featured['appliances_lag_2h'] = original_appliances.shift(12)
        
        # Create rolling window features
        df_featured['T_out_rolling_mean_1h'] = df_featured['T_out'].rolling(window=6).mean()
        df_featured['appliances_rolling_mean_1h'] = original_appliances.rolling(window=6).mean()
        
        # Drop rows with NaN values created by shift() and rolling()
        df_featured.dropna(inplace=True)
        
        print("Feature engineering complete.")
        return df_featured
