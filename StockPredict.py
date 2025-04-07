import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import argparse
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self, ticker_symbol, period="5y"):
        self.ticker = yf.Ticker(ticker_symbol)
        self.period = period
        self.scaler = StandardScaler()
        self.models = {}
        self.weights = {}
        self.dates = None
        self.current_price = None
        
    def prepare_data(self):
        df = self.ticker.history(period=self.period)
        self.dates = df.index  
        self.current_price = df['Close'].iloc[-1]  
        df.drop(['High', 'Low', 'Volume', 'Dividends', 'Stock Splits'], axis=1, inplace=True)
        
        data_o = df[['Open']].to_numpy()
        data_c = df[['Close']].to_numpy()
        
        X = np.zeros((len(data_o)-10, 3))
        Y = np.zeros(len(data_o)-10)
        
        for i in range(len(data_o)-10):
            X[i] = [data_o[i][0], data_c[i][0], data_o[i+1][0]]
            Y[i] = data_o[i+2][0]
            
        X_scaled = self.scaler.fit_transform(X)
        
        return train_test_split(X_scaled, Y, test_size=0.33, random_state=42)
    
    def get_date(self, index):
        return self.dates[index].strftime('%Y-%m-%d')
    
    def build_models(self):
        self.models = {
            'linear': LinearRegression(),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'neural_net': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'knn': KNeighborsRegressor(n_neighbors=5),
            'svm': SVR(kernel='rbf'),
            'gradient_boost': GradientBoostingRegressor(random_state=42)
        }
        
    def train_models(self, X_train, y_train):
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            results[name] = np.mean(cv_scores)
        
        total_score = sum(results.values())
        self.weights = {name: score/total_score for name, score in results.items()}
        
    def predict(self, X):
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        final_prediction = np.zeros(len(X))
        for name, pred in predictions.items():
            final_prediction += self.weights[name] * pred
            
        return final_prediction
    
    def predict_future(self, days_ahead=30):
        """Predict future prices for the next specified number of days"""
        latest_data = np.array([[self.current_price, self.current_price, self.current_price]])
        latest_data_scaled = self.scaler.transform(latest_data)
        
        future_predictions = []
        future_dates = []
        
        for i in range(days_ahead):
            next_day_pred = self.predict(latest_data_scaled)[0]
            future_predictions.append(next_day_pred)
            
            next_date = self.dates[-1] + timedelta(days=i+1)
            future_dates.append(next_date.strftime('%Y-%m-%d'))
            
            latest_data = np.array([[latest_data[0][1], latest_data[0][2], next_day_pred]])
            latest_data_scaled = self.scaler.transform(latest_data)
        
        return future_dates, future_predictions
