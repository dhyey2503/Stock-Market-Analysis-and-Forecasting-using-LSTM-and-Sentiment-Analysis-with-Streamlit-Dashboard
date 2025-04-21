import numpy as np
np.NaN = np.nan  # 👈 Patch BEFORE importing pandas_ta

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_autorefresh import st_autorefresh
import pandas_ta as ta

# --- Streamlit Page Config ---
st.set_page_config(page_title="📈 Dhyey's Stock Dashboard", page_icon="📉", layout="wide")
st.title("📈 Dhyey's Stock Market Dashboard with Sentiment Analysis")

# --- Sidebar Configuration ---
st.sidebar.header("Customize Your View 🎯")
theme_mode = st.sidebar.radio("Choose Theme Mode:", ["Light", "Dark"])

ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=180))
end_date = st.sidebar.date_input("End Date", datetime.today())

# Theme colors
if theme_mode == "Dark":
    bg_color = "#0e1117"
    text_color = "#fafafa"
    chart_bg = "#1f2937"
else:
    bg_color = "#ffffff"
    text_color = "#000000"
    chart_bg = "#ffffff"

# Apply custom styling
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .stText, .stMarkdown, .stDataFrame, .stChart {{
        color: {text_color};
    }}
    </style>
""", unsafe_allow_html=True)

# Auto-refresh every 5 minutes
st_autorefresh(interval=300000, key="refresh")

# --- Fetch Stock Data ---
data = yf.download(ticker, start=start_date, end=end_date)

# Clean MultiIndex if needed
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Calculate Moving Averages
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# Drop NaNs for plotting
data = data.dropna()

# --- Stock Price Line Chart ---
st.subheader(f"📈 {ticker} Stock Price with 20 & 50 Day Moving Averages")
st.line_chart(data[['Close', 'MA20', 'MA50']])

# --- Volume Chart ---
st.subheader(f"📈 {ticker} Trading Volume")
st.bar_chart(data['Volume'])

# --- Sentiment Analysis ---
st.subheader("🗾️ Sentiment Analysis on News Headlines")
analyzer = SentimentIntensityAnalyzer()

# Sample fallback news headlines
sample_news = [
    "Apple releases new product and sees rise in stock price.",
    "Investors worry about inflation and market volatility.",
    "Positive earnings report boosts investor confidence."
]

uploaded_file = st.file_uploader("Upload a TXT file with news headlines (one per line)", type="txt")
if uploaded_file:
    sample_news = uploaded_file.read().decode('utf-8').split('\n')

# Display sentiment scores
compound_scores = []
for i, news in enumerate(sample_news):
    if news.strip():
        score = analyzer.polarity_scores(news)
        compound_scores.append(score['compound'])
        st.write(f"📰 **News {i+1}:** {news}")
        st.write(f"   → Sentiment Score: `{score['compound']}`")

# --- Sentiment Trend Prediction (7 Days) ---
st.subheader("🔢 Sentiment Trend Prediction (Last 7 Days)")
# Simulated news headlines by day (can replace with real data)
news_by_day = {
    date.today() - timedelta(days=i): [f"Simulated headline {i}"]
    for i in range(7)
}

daily_sentiments = []
for news_date, headlines in news_by_day.items():
    scores = [analyzer.polarity_scores(h)['compound'] for h in headlines if h.strip()]
    if scores:
        daily_sentiments.append((news_date, sum(scores) / len(scores)))

df_sentiment = pd.DataFrame(daily_sentiments, columns=["Date", "Sentiment"]).sort_values("Date")
st.line_chart(df_sentiment.set_index("Date"))

# Predict trend
delta = df_sentiment["Sentiment"].iloc[-1] - df_sentiment["Sentiment"].iloc[0]
if delta > 0.1:
    trend = "📈 Improving (Positive Trend)"
elif delta < -0.1:
    trend = "📉 Declining (Negative Trend)"
else:
    trend = "➖ Stable / Neutral"
st.success(f"**Predicted Sentiment Trend:** {trend}")

# --- Show Last 10 Days of Data ---
if st.checkbox("Show Last 10 Days Raw Data 📄"):
    st.dataframe(data.tail(10))

# --- Download CSV ---
csv = data.tail(10).to_csv().encode('utf-8')
st.download_button(
    label="Download Last 10 Days Data as CSV",
    data=csv,
    file_name=f'{ticker}_last10days.csv',
    mime='text/csv',
)

# --- Technical Indicators ---
data['RSI'] = ta.rsi(data['Close'], length=14)
macd = ta.macd(data['Close'])
data['MACD_Line'] = macd['MACD_12_26_9']
data['MACD_Signal'] = macd['MACDs_12_26_9']
data['MACD_Hist'] = macd['MACDh_12_26_9']

# --- Technical Indicators Charts ---
st.subheader("📉 Technical Indicators")
st.write("🔹 **RSI (14-Day)**")
st.line_chart(data['RSI'])

st.write("🔹 **MACD (12, 26, 9)**")
st.line_chart(data[['MACD_Line', 'MACD_Signal', 'MACD_Hist']])

# --- Footer ---
st.markdown("""---""")
st.caption("Built by Dhyey Shah 🚀")