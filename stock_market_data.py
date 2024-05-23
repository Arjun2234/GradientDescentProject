import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from constant import *
import logging


class GradientDescent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add formatter to ch
        ch.setFormatter(formatter)

        # Add ch to logger
        self.logger.addHandler(ch)

    def load_data(self, file_path):
        try:
            """
            Load the CSV file and preprocess the data.
            Assumes the CSV file has columns 'Date', 'Closing_Price', and 'Volume'.
            """
            df = pd.read_csv(file_path)
            df = df.head(100)  # Use only the first 100 rows
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            # Check for NaN values and drop rows with NaN values
            nan_summary = df.isna().sum()
            if nan_summary.sum() > 0:
                print("NaN values found in the following columns:")
                print(nan_summary[nan_summary > 0])
                df = df.dropna()
                print("Rows with NaN values have been dropped.")

            df, closing_price_scaler = self.preprocess_data(df)
            return df, closing_price_scaler
        except Exception as e:
            self.logger.error(f"Error occurred while loading data: {e}")
            return None, None

    @staticmethod
    def preprocess_data(df):
        try:
            """
            Preprocess the data by scaling the 'Closing_Price'.
            """
            df = df.drop(columns='Volume')
            closing_price_scaler = StandardScaler()
            df['Closing_Price_scaled'] = closing_price_scaler.fit_transform(df[['Closing_Price']])
            return df, closing_price_scaler
        except Exception as e:
            logging.error(f"Error occurred during data preprocessing: {e}")
            return None, None

    @staticmethod
    def create_regression(X, y, learning_rate=0.0000001, epochs=1000):
        try:
            """
            Apply gradient descent to find the optimal slope and intercept for linear regression.
            Parameters:
            X : numpy array
                Independent variable (e.g., time steps).
            y : numpy array
                Dependent variable (scaled 'Closing_Price').
            learning_rate : float
                Step size for gradient descent updates.
            epochs : int
                Number of passes through the entire dataset.
            """
            m = 0.0  # Initial slope
            b = 0.0  # Initial intercept
            n = len(y)  # Number of data points

            for epoch in range(epochs):
                # Predicted values based on current m and b
                y_pred = m * X + b

                # Check for NaN or infinite values
                if np.isnan(y_pred).any() or np.isinf(y_pred).any() or np.isnan(m) or np.isnan(b) or np.isinf(
                        m) or np.isinf(b):
                    print("NaN or infinite values encountered. Please check your input data and parameters.")
                    return np.nan, np.nan

                # Compute the gradients
                gradient_m = (-2 / n) * np.sum(X * (y - y_pred))
                gradient_b = (-2 / n) * np.sum(y - y_pred)

                # Update parameters using gradient descent
                m -= learning_rate * gradient_m
                b -= learning_rate * gradient_b

                # Print debugging information
                if epoch % 100 == 0:
                    print(f"Epoch: {epoch}, Slope m: {m}, Intercept b: {b}")

            return m, b
        except Exception as e:
            logging.error(f"Error occurred during gradient descent: {e}")
            return np.nan, np.nan

    def plot_regression_line(self, X, y, m, b):
        try:

            """
            Plot the regression line against the data points.
    
            Parameters:
            X : numpy array
                Independent variable (e.g., time steps).
            y : numpy array
                Dependent variable (scaled 'Closing_Price').
            m : float
                Slope of the regression line.
            b : float
                Intercept of the regression line.
            """
            plt.scatter(X, y, color='blue')
            plt.plot(X, m * X + b, color='red')
            plt.xlabel('Time Steps')
            plt.ylabel('Scaled Closing Price')
            plt.title('Stock Market Prediction using Gradient Descent')
            plt.show()
        except Exception as e:
            logging.error(f"Error occurred in calling_function: {e}")

    def calling_function(self):
        gd = GradientDescent()
        # Load and preprocess data
        df, closing_price_scaler = gd.load_data(file_path)
        X = np.arange(len(df)).reshape(-1, 1)

        # Normalize X
        X_scaler = MinMaxScaler()
        X_scaled = X_scaler.fit_transform(X)

        y = df['Closing_Price_scaled'].values

        # Apply gradient descent to find the slope and intercept
        m, b = gd.create_regression(X_scaled.flatten(), y, learning_rate=0.0000001, epochs=1000)
        if not np.isnan(m) and not np.isnan(b):
            print("Slope m:", m)
            print("Intercept b:", b)


GradientDescent().calling_function()
