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

class TradingBot:
    def __init__(self, initial_budget=10000, initial_stocks=0):
        self.budget = initial_budget
        self.stocks = initial_stocks
        self.transaction_history = []
        
    def execute_trade(self, current_price, predicted_price, date, confidence_threshold=0.05):
        price_change = (predicted_price - current_price) / current_price
        
        if price_change > confidence_threshold and self.budget >= current_price:
            shares_to_buy = min(int(self.budget / current_price), 5)  
            cost = shares_to_buy * current_price
            self.budget -= cost
            self.stocks += shares_to_buy
            self.transaction_history.append({
                'type': 'buy',
                'shares': shares_to_buy,
                'price': current_price,
                'total': cost,
                'date': date
            })
            
        elif price_change < -confidence_threshold and self.stocks > 0:
            shares_to_sell = min(self.stocks, 5)  
            revenue = shares_to_sell * current_price
            self.budget += revenue
            self.stocks -= shares_to_sell
            self.transaction_history.append({
                'type': 'sell',
                'shares': shares_to_sell,
                'price': current_price,
                'total': revenue,
                'date': date
            })
            
    def get_portfolio_value(self, current_price):
        return self.budget + (self.stocks * current_price)
    
    def get_trading_recommendation(self, current_price, predicted_price, confidence_threshold=0.05):
        price_change = (predicted_price - current_price) / current_price
        
        if price_change > confidence_threshold:
            return "BUY", price_change
        elif price_change < -confidence_threshold:
            return "SELL", price_change
        else:
            return "HOLD", price_change
