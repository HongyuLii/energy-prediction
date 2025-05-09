import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

### Preprocessing
# Prepare the features and the label. We have two timeseries: sced (real-time) system lambda and day-ahead (DAP) system lambda. We want to predict the real-time system lambda for certain horizon (e.g. 6 hours ahead = 72 points in 5-mins). DAP are available more than 24 hours ahead.  

class DataProcessorUpdated:
    def __init__(self, self_back_window = 7*24, prediction_window = 24, sl_path = '../datasset/ercot_sl_2019_2023.csv', dap_path = '../datasset/hourly_ercot_day_ahead_sl_2019_2023.csv', fuel_path = '../datasset/ercot_fuel_2019_2023.csv', load_path = '../datasset/ercot_load_2019_2023.csv'):
        self.sl = None
        self.dap = None
        self.x_train_lstm = None
        self.y_train_lstm = None
        self.x_val_lstm = None
        self.y_val_lstm = None
        self.x_test_lstm = None
        self.y_test_lstm = None
        self.x_train_df_reg = None
        self.look_back_window = self_back_window
        self.prediction_window = prediction_window
        self.df = None
        self.sl_path = sl_path
        self.dap_path = dap_path
        self.fuel_path = fuel_path
        self.load_path = load_path
        # Standardization Factors
        self.y_mean_reg = None
        self.y_std_reg = None

    
    def clean_and_resample_data(self, file_path, time_col, freq='60min'):
            """
            Load, clean, and resample time-series data.
            :param file_path: Path to the CSV file.
            :param time_col: Name of the timestamp column.
            :param freq: Resampling frequency (default: '60min').
            :return: Cleaned DataFrame with consistent timestamps.
            """
            try:
                df = pd.read_csv(file_path)
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)

                # Resample to the desired frequency
                df = df.resample(freq).ffill()

                # Ensure continuous date range
                start_time = df.index.min()
                end_time = df.index.max()
                print(end_time)

                # Adjust end_time based on the frequency
                if freq == '5min' and end_time.minute % 5 != 0:
                    # For 5min frequency, adjust to the nearest 5-minute mark
                    remainder = end_time.minute % 5
                    end_time = end_time + pd.Timedelta(minutes=(5 - remainder))
                    end_time = end_time.replace(second=0)
                elif freq in ['1h', '60min'] and end_time.minute != 0:
                    # For 1h or 60min frequency, adjust to the nearest hour
                    end_time = end_time.replace(minute=0, second=0) + pd.Timedelta(hours=1)
                    
                date_range = pd.date_range(start=start_time, end=end_time, freq=freq)
                df = df.reindex(date_range)

                # Interpolate missing values
                df.interpolate(method='time', inplace=True)
                df.dropna(inplace=True)

                return df
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                return None
        
    def load_and_clean_data(self):
        self.sl = self.clean_and_resample_data(self.sl_path, 'sced_time_stamp_local')
        self.dap = self.clean_and_resample_data(self.dap_path, 'timestamp')
        self.fuel = self.clean_and_resample_data(self.fuel_path, 'interval_start_local')
        self.load = self.clean_and_resample_data(self.load_path, 'interval_start_local')

    
    def combine_all_data(self):
        # Work on copies to avoid modifying originals
        sl = self.sl.copy()
        dap = self.dap.copy()
        fuel = self.fuel.copy()
        load = self.load.copy()

        # Keep all columns but rename to avoid clashes
        dap.columns = [f'DAP_{col}' for col in dap.columns]
        sl.columns = [f'SCED_{col}' for col in sl.columns]
        fuel.columns = [f'Fuel_{col}' for col in fuel.columns]
        load.columns = [f'Load_{col}' for col in load.columns]

        # Combine
        self.df = pd.concat([dap, sl, fuel, load], axis=1)
        self.df['delta_price'] = self.df['SCED_system_lambda'] - self.df['DAP_SystemLambda']
    
    def shift_dap(self):
        self.df['DAP_SystemLambda'] = self.df['DAP_SystemLambda'].shift(-24)

    def unshift_dap(self):
        self.df['DAP_SystemLambda'] = self.df['DAP_SystemLambda'].shift(24)

    def split_data(self, shift_dap = True, feature_columns = None, target_col='SCED_system_lambda'):
        if self.df is None:
            print("Data not loaded properly. Please run `load_and_clean_data()` or assign to self.df.")
            return

        df = self.df.copy()
        if shift_dap:
            self.shift_dap()
            df = self.df.copy()
            self.unshift_dap()
        
        # Drop rows with any NaNs
        df.dropna(inplace=True)

        if feature_columns is None:
            feature_columns = [col for col in df.columns]

        self.x_train_df_reg = df.loc[:'2021-12-31 23:00:00', feature_columns]
        self.x_val_df_reg = df.loc['2022-01-01 00:00:00':'2022-12-24 23:00:00', feature_columns]
        self.x_test_df_reg = df.loc['2022-12-25 00:00:00':, feature_columns]

        self.y_train_df_reg = self.x_train_df_reg[[target_col]]
        self.y_val_df_reg = self.x_val_df_reg[[target_col]]
        self.y_test_df_reg = self.x_test_df_reg[[target_col]]
    
    # def apply_log_transform(self, target_col='SCED_system_lambda'):
    #     """
    #     Apply log(x + 1 - min) to all numeric columns in x_train/x_val/x_test.
    #     Optionally apply to y_train/y_val/y_test.
    #     """
    #     self.log_min_vals = {}
    #     # Feature sets
    #     for df in [self.x_train_df_reg, self.x_val_df_reg, self.x_test_df_reg]:
    #         for col in df.select_dtypes(include='number').columns:
    #             if col not in self.log_min_vals:
    #                 min_val = self.x_train_df_reg[col].min()
    #                 self.log_min_vals[col] = min_val
    #             else:
    #                 min_val = self.log_min_vals[col]
    #             df[col] = np.log(df[col] + 1 - min_val)
                
    #     self.y_train_df_reg = self.x_train_df_reg[['SCED_system_lambda']]
    #     self.y_val_df_reg = self.x_val_df_reg[['SCED_system_lambda']]
    #     self.y_test_df_reg = self.x_test_df_reg[['SCED_system_lambda']]
    
    # def inverse_log_transform(self, data, target_col='SCED_system_lambda'):
    #     """
    #     Reverse the log(x + 1 - min_val) transformation using stored min_val.
    #     :param data: A NumPy array, Pandas Series, or DataFrame column.
    #     :param column: The column name to look up min_val.
    #     :return: The inverse-transformed data.
    #     """
    #     if target_col not in self.log_min_vals:
    #         raise ValueError(f"Min value for '{target_col}' not found. Did you apply log transform?")
        
    #     min_val = self.log_min_vals[target_col]
    #     return np.exp(data) + min_val - 1

    # def apply_gaussian_transform(self):
    #     """
    #     Apply quantile-based Gaussian normalization to features and target.
    #     Stores transformers for inverse transform later.
    #     """
    #     self.x_gauss_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
    #     self.y_gauss_transformer = QuantileTransformer(output_distribution='normal', random_state=0)

    #     self.x_train_reg = pd.DataFrame(
    #         self.x_gauss_transformer.fit_transform(self.x_train_df_reg),
    #         index=self.x_train_df_reg.index,
    #         columns=self.x_train_df_reg.columns
    #     )
    #     self.x_val_reg = pd.DataFrame(
    #         self.x_gauss_transformer.transform(self.x_val_df_reg),
    #         index=self.x_val_df_reg.index,
    #         columns=self.x_val_df_reg.columns
    #     )
    #     self.x_test_reg = pd.DataFrame(
    #         self.x_gauss_transformer.transform(self.x_test_df_reg),
    #         index=self.x_test_df_reg.index,
    #         columns=self.x_test_df_reg.columns
    #     )

    #     self.y_train_reg = pd.DataFrame(
    #         self.y_gauss_transformer.fit_transform(self.y_train_df_reg),
    #         index=self.y_train_df_reg.index,
    #         columns=self.y_train_df_reg.columns
    #     )
    #     self.y_val_reg = pd.DataFrame(
    #         self.y_gauss_transformer.transform(self.y_val_df_reg),
    #         index=self.y_val_df_reg.index,
    #         columns=self.y_val_df_reg.columns
    #     )
    #     self.y_test_reg = pd.DataFrame(
    #         self.y_gauss_transformer.transform(self.y_test_df_reg),
    #         index=self.y_test_df_reg.index,
    #         columns=self.y_test_df_reg.columns
    #     )
    
    # def inverse_gaussian_transform_y(self, y_pred):
    #     y_pred_real = np.vstack([
    #         self.y_gauss_transformer.inverse_transform(y_pred[:, i].reshape(-1, 1)).flatten()
    #         for i in range(y_pred.shape[1])
    #     ]).T 
    #     return y_pred_real

    def standardize_data(self):
        # Standardization
        self.x_mean_reg, self.x_std_reg = self.x_train_df_reg.mean(), self.x_train_df_reg.std()
        self.y_mean_reg, self.y_std_reg = self.y_train_df_reg.mean(), self.y_train_df_reg.std()
        self.x_std_reg += 1e-5

        # Normalize features and targets
        self.x_train_reg = (self.x_train_df_reg - self.x_mean_reg) / self.x_std_reg
        self.x_val_reg = (self.x_val_df_reg - self.x_mean_reg) / self.x_std_reg
        self.x_test_reg = (self.x_test_df_reg - self.x_mean_reg) / self.x_std_reg

        self.y_train_reg = (self.y_train_df_reg - self.y_mean_reg) / self.y_std_reg
        self.y_val_reg = (self.y_val_df_reg - self.y_mean_reg) / self.y_std_reg
        self.y_test_reg = (self.y_test_df_reg - self.y_mean_reg) / self.y_std_reg
    
    def denormalization(self, data, target_col='SCED_system_lambda'):
        """
        Reverses standardization of model outputs using stored mean/std of the target column.
        """
        mean = self.y_mean_reg[target_col]
        std = self.y_std_reg[target_col]
        return data * std + mean

    def shift_data(self):
        # Create LSTM input-output sequences
        n_steps_in = self.look_back_window
        n_steps_out = self.prediction_window

        def create_sequences(x, y):
            x_seq = np.array([x[i:i + n_steps_in] for i in range(len(x) - n_steps_in - n_steps_out + 1)])
            y_seq = np.array([y[i + n_steps_in:i + n_steps_in + n_steps_out] for i in range(len(y) - n_steps_in - n_steps_out + 1)])
            return x_seq, y_seq

        self.x_train_lstm, self.y_train_lstm = create_sequences(self.x_train_reg.values, self.y_train_reg.values)
        self.x_val_lstm, self.y_val_lstm = create_sequences(self.x_val_reg.values, self.y_val_reg.values)
        self.x_test_lstm, self.y_test_lstm = create_sequences(self.x_test_reg.values, self.y_test_reg.values)
        
    def get_data(self):
        return self.df, self.x_train_lstm, self.y_train_lstm, self.x_val_lstm, self.y_val_lstm, self.x_test_lstm, self.y_test_lstm
    
    def get_reg_data(self):
        return self.y_std_reg, self.y_mean_reg
    
    def flatten(self):
        self.x_train_mlp = self.x_train_lstm.reshape(self.x_train_lstm.shape[0], -1)  # Flatten from (315349, 288, 2) to (315349, 576)
        self.x_val_mlp = self.x_val_lstm.reshape(self.x_val_lstm.shape[0], -1)
        self.x_test_mlp = self.x_test_lstm.reshape(self.x_test_lstm.shape[0], -1)

        # Reshape the targets (if needed)
        self.y_train_mlp = self.y_train_lstm.reshape(self.y_train_lstm.shape[0], -1)  # Flatten from (315349, 12, 1) to (315349, 12)
        self.y_val_mlp = self.y_val_lstm.reshape(self.y_val_lstm.shape[0], -1)
        self.y_test_mlp = self.y_test_lstm.reshape(self.y_test_lstm.shape[0], -1)

        return self.x_train_mlp, self.x_val_mlp, self.x_test_mlp, self.y_train_mlp, self.y_val_mlp, self.y_test_mlp

    def prepare_data(self):
        self.load_and_clean_data()
        self.combine_all_data()
        self.split_data()
        self.apply_log_transform()
        self.standardize_data()
        self.shift_data()

    def run(self):
        self.load_and_clean_data_sced()
        self.load_and_clean_data_dap()
        self.load_and_clean_data_fuel()
        self.load_and_clean_data_load()
        self.prepare_data()