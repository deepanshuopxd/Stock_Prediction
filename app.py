from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import requests
from textblob import TextBlob
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
from markupsafe import Markup
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['nm'].upper()

    def fetch_stock_data(symbol):
        end_date = datetime.now()
        start_date = datetime(end_date.year - 2, end_date.month, end_date.day)
        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty:
            raise ValueError("Invalid stock symbol or no data available.")
        df.reset_index(inplace=True)
        return df

    def arima_model(data):
        train_size = int(len(data) * 0.8)
        train, test = data[:train_size], data[train_size:]
        history = list(train)
        predictions = []
        for t in range(len(test)):
            model = ARIMA(history, order=(15, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast()[0]
            predictions.append(forecast)
            history.append(test[t])
        rmse = math.sqrt(mean_squared_error(test, predictions))
        return predictions[-1], predictions, rmse

    def linear_regression_model(data):
        forecast_out = 30
        data['CloseAfterN'] = data['Close'].shift(-forecast_out)
        X = data[['Close']].iloc[:-forecast_out]
        y = data['CloseAfterN'].iloc[:-forecast_out]
        X_forecast = data[['Close']].iloc[-forecast_out:]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_forecast_scaled = scaler.transform(X_forecast)

        model = LinearRegression()
        model.fit(X_scaled, y)
        forecast = model.predict(X_forecast_scaled)
        rmse = math.sqrt(mean_squared_error(y[-len(forecast):], forecast))
        return forecast[0], forecast, rmse

    def sentiment_analysis(symbol):
        API_KEY = '4g6ghyxq144urydocxz9rrscjdhufnv7xxwsw5p3' 
        url = f'https://stocknewsapi.com/api/v1?tickers={symbol}&items=3&token={API_KEY}'
        response = requests.get(url).json()

        if not response.get('data'):
            return 0, "Could not fetch news sentiment", []
      
        positive, negative , neutral  = 0, 0 ,0
        
        news_data = []
        for article in response['data']:
            sentiment = TextBlob(article['title']).sentiment.polarity
            news_data.append({
                'title': article['title'],
                'url': article.get('url'),
                'sentiment': 'Positive' if sentiment > 0.2 else 'Negative' if sentiment < -0.2 else 'Neutral',
                'content': article.get('content')
            })
            if sentiment > 0.2:
                positive += sentiment
            elif sentiment < -0.2:
                negative += abs(sentiment)
            else:
                neutral +=  abs(sentiment)


        overall_sentiment = positive - negative + neutral
        if overall_sentiment > 0.5:
            sentiment_text = "Positive"
        elif overall_sentiment < 0:
            sentiment_text = "Negative"
        else:
            sentiment_text = "Neutral"

        return overall_sentiment, sentiment_text, news_data

    def plot_to_mpld3(fig, x_data, y_data):
        scatter = ax.scatter(x_data, y_data, alpha=0)  
        labels = [f"Date: {x} Price: ${y}" for x, y in zip(x_data, y_data)]
        tooltip = plugins.PointLabelTooltip(scatter, labels)
        plugins.connect(fig, tooltip)
        mpld3_html = mpld3.fig_to_html(fig)
        plt.close(fig)
        return mpld3_html

    try:
        stock_data = fetch_stock_data(stock_symbol)
    except ValueError as e:
        return render_template('index.html', error=str(e))

    stock_data = stock_data[['Date', 'Close']].dropna()
    arima_pred, arima_predictions, arima_rmse = arima_model(stock_data['Close'].values)
    lr_pred, lr_predictions, lr_rmse = linear_regression_model(stock_data)

    current_price = stock_data['Close'].iloc[-1].item()
    sentiment_score, sentiment_text, news_articles = sentiment_analysis(stock_symbol)
    
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    fig, ax = plt.subplots()
    ax.plot(stock_data['Date'], stock_data['Close'], label='Stock Price', color='blue')
    ax.set_title(f"{stock_symbol} Stock Price")
    ax.set_xlabel('Date', fontsize=12, color='black')
    ax.set_ylabel('Close Price', fontsize=12, color='black')
    ax.tick_params(axis='x', labelsize=10, colors='black')
    ax.tick_params(axis='y', labelsize=10, colors='black')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    stock_plot = plot_to_mpld3(fig, stock_data['Date'].dt.strftime('%Y-%m-%d'), stock_data['Close'])

    fig, ax = plt.subplots()
    arima_dates = stock_data['Date'][-len(arima_predictions):]
    ax.plot(stock_data['Date'], stock_data['Close'], label='Original Price', color='blue')
    ax.plot(arima_dates, arima_predictions, label='ARIMA Predictions', linestyle='--', color='red')
    ax.set_title(f"{stock_symbol} ARIMA Predictions")
    ax.set_xlabel('Date', fontsize=12, color='black')
    ax.set_ylabel('Price', fontsize=12, color='black')
    ax.tick_params(axis='x', labelsize=10, colors='black')
    ax.tick_params(axis='y', labelsize=10, colors='black')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    arima_plot = plot_to_mpld3(fig, arima_dates.dt.strftime('%Y-%m-%d'), arima_predictions)

    fig, ax = plt.subplots()
    lr_dates = stock_data['Date'][-len(lr_predictions):]
    ax.plot(stock_data['Date'], stock_data['Close'], label='Original Price', color='blue')
    ax.plot(lr_dates, lr_predictions, label='LR Predictions', linestyle='--', color='green')
    ax.set_title(f"{stock_symbol} Linear Regression Predictions")
    ax.set_xlabel('Date', fontsize=12, color='black')
    ax.set_ylabel('Price', fontsize=12, color='black')
    ax.tick_params(axis='x', labelsize=10, colors='black')
    ax.tick_params(axis='y', labelsize=10, colors='black')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    lr_plot = plot_to_mpld3(fig, lr_dates.dt.strftime('%Y-%m-%d'), lr_predictions)
    
    if sentiment_score  > 0.5:
        recommendation = "BUY"
    elif sentiment_score < 0:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    print(sentiment_score)
    return render_template('results.html',
                           stock_symbol=stock_symbol,
                           current_price=round(current_price, 2),
                           arima_pred=round(arima_pred, 2),
                           lr_pred=round(lr_pred, 2),
                           arima_rmse=round(arima_rmse, 2),
                           lr_rmse=round(lr_rmse, 2),
                           sentiment_text=sentiment_text,
                           recommendation=recommendation,
                           stock_plot=Markup(stock_plot),
                           arima_plot=Markup(arima_plot),
                           lr_plot=Markup(lr_plot),
                           news_articles=news_articles)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
