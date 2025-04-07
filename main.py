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

def main():
    parser = argparse.ArgumentParser(description='Stock Prediction and Trading Bot')
    parser.add_argument('--ticker', type=str, default='HUM',
                      help='Stock ticker symbol (default: HUM)')
    parser.add_argument('--budget', type=float, default=10000,
                      help='Initial budget for trading (default: 10000)')
    parser.add_argument('--stocks', type=int, default=0,
                      help='Initial number of stocks (default: 0)')
    parser.add_argument('--future-days', type=int, default=30,
                      help='Number of days to predict into the future (default: 30)')
    args = parser.parse_args()

    predictor = StockPredictor(args.ticker)
    bot = TradingBot(initial_budget=args.budget, initial_stocks=args.stocks)
    
    print(f"\nAnalyzing stock: {args.ticker}")
    print(f"Initial budget: ${args.budget:.2f}")
    print(f"Initial stocks: {args.stocks}")
    
    X_train, X_test, y_train, y_test = predictor.prepare_data()
    predictor.build_models()
    predictor.train_models(X_train, y_train)
    
    train_predictions = predictor.predict(X_train)
    test_predictions = predictor.predict(X_test)
    
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    
    print(f"\nModel Performance:")
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Testing MSE: {test_mse:.2f}")
    
    for i in range(len(X_test)):
        current_price = y_test[i]
        predicted_price = test_predictions[i]
        date = predictor.get_date(len(X_train) + i)
        bot.execute_trade(current_price, predicted_price, date)
    
    final_value = bot.get_portfolio_value(y_test[-1])
    print(f"\nHistorical Trading Results:")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Cash: ${bot.budget:.2f}")
    print(f"Stocks: {bot.stocks}")
    
    print("\nHistorical Transaction History:")
    for transaction in bot.transaction_history:
        print(f"{transaction['date']} - {transaction['type'].upper()}: {transaction['shares']} shares at ${transaction['price']:.2f}")
    
    future_dates, future_predictions = predictor.predict_future(days_ahead=args.future_days)
    
    current_price = predictor.current_price
    print(f"\nCurrent price of {args.ticker}: ${current_price:.2f}")
    print("\nFuture Price Predictions and Trading Recommendations:")
    
    for i in range(len(future_dates)):
        date = future_dates[i]
        predicted_price = future_predictions[i]
        action, change = bot.get_trading_recommendation(current_price, predicted_price)
        
        print(f"{date} - Predicted Price: ${predicted_price:.2f} | Recommendation: {action} (Change: {change:.2%})")
        
        current_price = predicted_price

if __name__ == "__main__":
    main()
