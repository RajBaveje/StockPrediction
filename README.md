# Future Stock Price Prediction AI and Trading Bot

## Author
- Raj Baveje

## Overview
This is a Python Trading and Stock Prediction Bot using past stock data predicting future prices and making trading decisions. Various machine learning models are used to provide predictions and the predictions of the models are then combined using weighted ensemble methods. Buy/sell/hold recommendations are tested and simulated trades are placed based on predictions for the movement of stock prices.
The project has two main components:
A Stock Predictor, which creates historical data, trains multiple models, and predicts future prices.
A Trading Bot, which takes simulated trading decisions based on the output of the predictor, keeps a transaction history, and offers portfolio performance reports.

The tool takes command-line arguments for customization and offers insights such as model performance, trading recommendations, and final portfolio value.

## Project Structure
### StockPredictor
- Downloads historical data through the yfinance API
- Cleans and formats data for supervised learning
- Trains multiple regression models (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, MLPRegressor, KNeighborsRegressor, SVR, GradientBoostingRegressor)
- Combines model predictions with a weighted ensemble approach
- Generates short-term (30-day) future price predictions

### TradingBot
- Simulates trades based on predicted vs. actual price change
- Uses a simple confidence threshold to determine buy/sell/hold actions
- Tracks budget, stock holdings, and transaction history
- Calculates total portfolio value

### Main Function
- Parses command-line options (e.g., ticker symbol, initial capital)
- Trains models on history and evaluates model performance
- Simulates  trading off of test data
- Predicts and recommends trades for a window of future time
- Outputs summary statistics and history of transactions

## How to Compile and Run
### Install Dependencies
- pip install yfinance, pandas, numpy, scikit-learn
### Run Script
- python stock_bot.py --ticker TSLA --budget 5000 --stocks 5 --future-days 50






