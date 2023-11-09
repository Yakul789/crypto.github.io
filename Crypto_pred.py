import numpy as np
import seaborn as sns
import yfinance as yf
import streamlit as st
import pandas as pd
import datetime
from prophet import Prophet
import streamlit as st
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from bs4 import BeautifulSoup
from textblob import TextBlob
from googlesearch import search
import requests

# Set the default page to be the front page
st.set_page_config(page_title="Crypto Forecast & News Sentiment", page_icon="✅")

#interval of forecasting(From 1st Jan 2020 to now)
start = '2020-01-01'
current = datetime.datetime.now()
# Structure for the web application
st.title('Cryptocurrency Forecast & News Sentiment Analysis')
# Mapping Tickers to Crypto currencies


# Mapping Tickers to Crypto currencies
crypto_mapping = {
   'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'DOGE-USD': 'Dogecoin',
    'BNB-USD': 'Binance Coin',
    'ADA-USD': 'Cardano',
    'SOL-USD': 'Solana',
    'XRP-USD': 'Ripple',
    'DOT-USD': 'Polkadot',
    'LTC-USD': 'Litecoin',
    'BCH-USD': 'Bitcoin Cash'
}

# Reverse the dictionary to map names to tickers
reverse_mapping = {v: k for k, v in crypto_mapping.items()}

# Create a select box with cryptocurrency names
selected_name = st.selectbox("Select a cryptocurrency", list(crypto_mapping.values()))

# Use the selected name to fetch the corresponding cryptocurrency ticker
selected_ticker = reverse_mapping.get(selected_name)

if selected_ticker is not None:
    st.write(f"The ticker symbol for {selected_name} is {selected_ticker}")
else:
    st.write(f"No ticker symbol found for {selected_name}")

prediction_unit = st.selectbox('Prediction Unit', ['Months', 'Weeks', 'Days'])
n_units = st.number_input(f'Number of {prediction_unit}', min_value=1)

if prediction_unit == 'Months':
    period = n_units * 30  # Assuming 30 days in a month
elif prediction_unit == 'Weeks':
    period = n_units * 7  # 7 days in a week
else:  # prediction_unit == 'Days'
    period = n_units

# Define the data loading function using the selected ticker
@st.cache_data  # Cache the searched data
def data_load(crypto):
    data = yf.download(selected_ticker, start, current)
    data.reset_index(inplace=True)
    return data

# Plot the current data in table format
data = data_load(selected_ticker)
st.subheader('Current trend')
st.write(data.tail())


#plot current trend of closing prices
candlestick_trace = go.Candlestick(
    x=data['Date'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
)
#barplot for plotting current trade volume data
volume_trace = go.Bar(
    x=data['Date'],
    y=data['Volume'],
    name='Volume',
    marker=dict(color='blue')  # Customize bar color
)

#use declared functions to plot the visualizations
def plot_current():
    fig_1 = go.Figure(data=[candlestick_trace])
    fig_1.layout.update(title_text='Closing Price Trend', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_1)

    fig_2 = go.Figure(data=[volume_trace])
    fig_2.layout.update(title_text='Trade Volume Trend', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_2)
#call the function
plot_current()

#prepare train data for forecast
#use the date and close columns
#we will try forecasting closing prices 
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

#we will use the prophet model
#create an object of prophet class and feed the training data to it
mod = Prophet()
mod.fit(df_train)
future = mod.make_future_dataframe(periods=period)#makes the future forecasted dataframe 
forecast = mod.predict(future)#makes the predictions

#plot forecasted data in dataframe format
st.subheader('Forecast data')
st.write(forecast.tail())

#plot line plot for forecast
fig1 = plot_plotly(mod, forecast)
st.plotly_chart(fig1)

#plot weekly and yearly trend components
st.write('Forecast components')
fig2 = mod.plot_components(forecast)
st.write(fig2)

def get_historical_stock_data(stock_symbol, start_date, end_date):##get stock news history
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data

def extract_news_links(search_query, num_links=5):##use news links to extract articles
    news_links = []
    search_results = search(search_query, num_results=num_links, lang='en')

    for result in search_results:
        if "news" in result:##use news as keyword for search
            news_links.append(result)
    return news_links

def analyze_sentiment(text):##news analysis using textblob
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score##get sentiment scorees between -1 and 1

def main():##main function for news analysis
    start_date = '2020-01-01'  
    end_date = datetime.datetime.now()

    st.title("Recent Articles")

    try:
        stock_data = get_historical_stock_data(selected_ticker, start_date, end_date)
        if stock_data.empty:##if no stock data found then display required message
            st.write("No stock data found for the given date range.")
            return

        search_query = f"{selected_name}  news"##display news for selected cryptocurrency
        news_links = extract_news_links(search_query)

        if not news_links:
            st.write("No news articles found.")##if no articles found then display required message
            return

        st.write(f"Analyzing news articles related to {selected_name} stock:")##title

        for link in news_links:##interact with news links and display the news articles 
            try:
                response = requests.get(link)
                soup = BeautifulSoup(response.text, 'html.parser')##use beautifulsoup library to scrape news data from html/xml webpages
                news_text = ""
                for paragraph in soup.find_all('p'):
                    news_text += paragraph.get_text()
                sentiment_score = analyze_sentiment(news_text)#display news analysis

                st.write(f"News Article Link: {link}")##display the article link
                st.write(f"Sentiment Score: {sentiment_score}")
                st.write("-" * 50)
            except Exception as e:##handle exceptions
                st.write(f"Error processing article: {e}")
    except Exception as e:
        st.write(f"Error: {e}")

if __name__ == "__main__":
    main()