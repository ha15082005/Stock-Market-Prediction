import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.title('Stock Price Predictor')
st.sidebar.caption('Adjust your settings:')


# Main function to control app options
def main():
    choice = st.sidebar.selectbox('Choose an Action', ['View Indicators', 'Show Recent Data', 'Make Predictions'])
    if choice == 'View Indicators':
        show_indicators()
    elif choice == 'Show Recent Data':
        show_data()
    else:
        make_predictions()


# Download stock data
@st.cache_resource
def get_data(stock_symbol, start, end):
    return yf.download(stock_symbol, start=start, end=end, progress=False)

# Sidebar input for stock selection and date range
symbol = st.sidebar.text_input('Stock Symbol', value='AAPL').upper()
today = datetime.date.today()
duration_days = st.sidebar.number_input('Days to Go Back', value=5000)
before = today - datetime.timedelta(days=duration_days)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)

if st.sidebar.button('Submit'):
    if start_date < end_date:
        st.sidebar.success('Start date: %s\n\nEnd date: %s' % (start_date, end_date))
        download_data(symbol, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')

data = get_data(symbol, start_date, end_date)
scaler = StandardScaler()

# Function to show technical indicators
def show_indicators():
    st.header('Technical Indicators')
    indicator = st.radio('Select an Indicator', ['Close Price', 'MACD', 'RSI', 'SMA', 'EMA'])

    macd = MACD(data.Close.squeeze()).macd()
    rsi = RSIIndicator(data.Close.squeeze()).rsi()
    sma = SMAIndicator(data.Close.squeeze(), window=14).sma_indicator()
    ema = EMAIndicator(data.Close.squeeze()).ema_indicator()

    if indicator == 'Close Price':
        st.line_chart(data['Close'], color='#ff6961')
    elif indicator == 'MACD':
        st.line_chart(macd)
    elif indicator == 'RSI':
        st.line_chart(rsi, color='#5f9ea0')
    elif indicator == 'SMA':
        st.line_chart(sma, color='#ff6347')
    elif indicator == 'EMA':
        st.line_chart(ema, color='#9370db')


# Display recent data
def show_data():
    st.header('Recent Stock Data')
    st.dataframe(data.tail(10), width=800)


# Prediction function for stock prices
def make_predictions():
    model_choice = st.radio('Pick a Model',
                            ['Linear Regression', 'Random Forest', 'Extra Trees', 'K-Nearest Neighbors', 'XGBoost'])
    days_to_predict = st.number_input('Days to Predict', value=10, step=1)

    if st.button('Run Prediction'):
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Extra Trees': ExtraTreesRegressor(),
            'K-Nearest Neighbors': KNeighborsRegressor(),
            'XGBoost': XGBRegressor()
        }
        chosen_model = models[model_choice]
        run_model(chosen_model, days_to_predict)


# Helper function to train and make predictions
def run_model(model, days):
    df = data[['Close']].copy()
    df['Future'] = df['Close'].shift(-days)
    X = scaler.fit_transform(df[['Close']].values[:-days])
    y = df['Future'].values[:-days]
    X_forecast = scaler.transform(df[['Close']].values[-days:])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    st.write(f"R² Score: {r2_score(y_test, predictions)}")
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions)}")

    forecast = model.predict(X_forecast)
    for day, prediction in enumerate(forecast, start=1):
        st.write(f"Day {day}: {prediction}")


if __name__ == '__main__':
    main()
