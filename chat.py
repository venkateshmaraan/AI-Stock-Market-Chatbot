import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import re

st.set_page_config(
    page_title="StockSensi - Your AI Stock Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("StockSensi")
st.markdown("""
This AI assistant helps you navigate the stock market with ease. Ask questions in plain English about:
- Real-time stock prices
- Historical performance
- Price predictions
- Stock comparisons
- Investment recommendations
- Learning resources
""")


with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to:", ["Chat", "Market Overview", "Learning Center"])
    
    st.header("Settings")
    days_to_predict = st.slider("Prediction Days", 7, 30, 14)
    
    st.header("About")
    st.info("StockSensi uses Yahoo Finance data and machine learning to provide stock insights.")


def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('currentPrice', info.get('regularMarketPrice', None))
        if price:
            change = info.get('regularMarketChangePercent', 0)
            return price, change
        return None, None
    except:
        return None, None


def get_stock_history(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except:
        return None


def predict_stock(ticker, days=14):
    try:
        
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365)
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            return None, None
        
        
        df = data[['Close']].copy()
        df['Prediction'] = df['Close'].shift(-1)
        df.dropna(inplace=True)
        

        window_size = 30
        for i in range(1, window_size+1):
            df[f'Lag_{i}'] = df['Close'].shift(i)
        
        df.dropna(inplace=True)
        
        
        X = df.drop(['Close', 'Prediction'], axis=1)
        y = df['Prediction']
        
        
        model = LinearRegression()
        model.fit(X, y)
        
        
        last_window = df[['Close'] + [f'Lag_{i}' for i in range(1, window_size+1)]].iloc[-1:].copy()
        
        
        predictions = []
        for i in range(days):
            
            next_price = model.predict(last_window.drop(['Close'], axis=1))[0]
            predictions.append(next_price)
            
            
            for j in range(window_size, 1, -1):
                last_window[f'Lag_{j}'] = last_window[f'Lag_{j-1}']
            last_window[f'Lag_1'] = last_window['Close']
            last_window['Close'] = next_price
        
        
        last_date = data.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(days)]
        
        
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': predictions
        })
        pred_df.set_index('Date', inplace=True)
        
        return data, pred_df
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None


def compare_stocks(tickers, period="1y"):
    try:
        data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                data[ticker] = hist['Close']
        
        if data:
            df = pd.DataFrame(data)
            return df
        return None
    except:
        return None


def get_recommendations():

    recommendations = {
        'Growth': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
        'Dividend': ['JNJ', 'PG', 'KO', 'VZ', 'PFE'],
        'Value': ['BRK-B', 'JPM', 'BAC', 'WMT', 'DIS']
    }
    return recommendations


def extract_ticker(query):
    
    ticker_matches = re.findall(r'\b[A-Z]{1,5}\b', query)
    common_words = ['AI', 'ML', 'API', 'CEO', 'CFO', 'IPO', 'ETF']
    
    
    tickers = [t for t in ticker_matches if t not in common_words]
    
    
    pattern_matches = re.findall(r'(?:ticker|symbol|stock)[\s:]+([A-Z]{1,5})', query, re.IGNORECASE)
    if pattern_matches:
        tickers.extend(pattern_matches)
    
    return list(set(tickers))


def identify_intent(query):
    query = query.lower()
    
    if any(word in query for word in ['price', 'worth', 'value', 'cost', 'current']):
        return 'price'
    elif any(word in query for word in ['history', 'historical', 'past', 'performance', 'trend']):
        return 'history'
    elif any(word in query for word in ['predict', 'prediction', 'forecast', 'future', 'will']):
        return 'prediction'
    elif any(word in query for word in ['compare', 'versus', 'vs', 'against', 'difference']):
        return 'comparison'
    elif any(word in query for word in ['recommend', 'suggestion', 'best', 'top', 'should buy']):
        return 'recommendation'
    elif any(word in query for word in ['learn', 'tutorial', 'guide', 'how to', 'explain']):
        return 'learning'
    elif any(word in query for word in ['calculate', 'computation', 'ratio', 'metric', 'indicator']):
        return 'calculation'
    else:
        return 'general'


def process_query(query):
    intent = identify_intent(query)
    tickers = extract_ticker(query)
    
    if intent == 'price' and tickers:
        ticker = tickers[0]
        price, change = get_stock_price(ticker)
        if price:
            return f"The current price of {ticker} is ${price:.2f} ({change:.2f}%)."
        else:
            return f"Sorry, I couldn't fetch the current price for {ticker}."
    
    elif intent == 'history' and tickers:
        ticker = tickers[0]
        period = "1y"  # Default to 1 year
        if "month" in query.lower():
            period = "1mo"
        elif "year" in query.lower() and "5" in query:
            period = "5y"
        
        history = get_stock_history(ticker, period)
        if history is not None:
            st.subheader(f"{ticker} Historical Performance")
            st.line_chart(history['Close'])
            return f"Here's the historical performance of {ticker} for the requested period."
        else:
            return f"Sorry, I couldn't fetch historical data for {ticker}."
    
    elif intent == 'prediction' and tickers:
        ticker = tickers[0]
        days = days_to_predict  
        
        history, predictions = predict_stock(ticker, days)
        if history is not None and predictions is not None:
            
            st.subheader(f"{ticker} Price Prediction (Next {days} Days)")
            
           
            combined = pd.DataFrame({
                'Historical': history['Close'][-30:],  
                'Predicted': None
            })
            
            
            prediction_df = pd.DataFrame({
                'Historical': None,
                'Predicted': predictions['Predicted_Close']
            }, index=predictions.index)
            
            
            plot_df = pd.concat([combined, prediction_df])
            
            st.line_chart(plot_df)
            return f"Based on historical trends, here's my prediction for {ticker} over the next {days} days."
        else:
            return f"Sorry, I couldn't generate predictions for {ticker}."
    
    elif intent == 'comparison' and len(tickers) >= 2:
        
        period = "1y"  
        if "month" in query.lower():
            period = "1mo"
        elif "year" in query.lower() and "5" in query:
            period = "5y"
        
        comparison = compare_stocks(tickers, period)
        if comparison is not None:
            
            normalized = comparison.div(comparison.iloc[0]).multiply(100)
            
            st.subheader("Stock Comparison")
            st.line_chart(normalized)
            
            
            performance = {}
            for ticker in tickers:
                start_price = comparison[ticker].iloc[0]
                end_price = comparison[ticker].iloc[-1]
                percent_change = ((end_price - start_price) / start_price) * 100
                performance[ticker] = percent_change
            
            
            perf_df = pd.DataFrame({
                'Ticker': performance.keys(),
                'Change (%)': [f"{p:.2f}%" for p in performance.values()]
            })
            st.table(perf_df)
            
            
            best = max(performance, key=performance.get)
            return f"Here's the comparison between {', '.join(tickers)}. {best} has performed the best with a {performance[best]:.2f}% change."
        else:
            return f"Sorry, I couldn't compare those stocks. Please check the ticker symbols."
    
    elif intent == 'recommendation':
        recs = get_recommendations()
        
        st.subheader("Top Stock Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Growth Stocks")
            for ticker in recs['Growth']:
                price, change = get_stock_price(ticker)
                st.markdown(f"**{ticker}**: ${price:.2f} ({change:.2f}%)")
        
        with col2:
            st.markdown("### Dividend Stocks")
            for ticker in recs['Dividend']:
                price, change = get_stock_price(ticker)
                st.markdown(f"**{ticker}**: ${price:.2f} ({change:.2f}%)")
        
        with col3:
            st.markdown("### Value Stocks")
            for ticker in recs['Value']:
                price, change = get_stock_price(ticker)
                st.markdown(f"**{ticker}**: ${price:.2f} ({change:.2f}%)")
        
        return "Here are some stock recommendations across different investment strategies. Remember to do your own research before investing."
    
    elif intent == 'learning':
        st.subheader("Stock Market Learning Resources")
        
        resources = {
            "Beginner Guides": [
                "Understanding Stock Market Basics",
                "How to Read Stock Charts",
                "Fundamental vs Technical Analysis"
            ],
            "Intermediate Topics": [
                "Valuation Metrics Explained",
                "Building a Balanced Portfolio",
                "Risk Management Strategies"
            ],
            "Practice Tools": [
                "Virtual Trading Simulator",
                "Market Analysis Exercises",
                "Investment Case Studies"
            ]
        }
        
        for category, items in resources.items():
            st.markdown(f"### {category}")
            for item in items:
                st.markdown(f"- {item}")
        
        return "Here are some learning resources to help you understand the stock market better. What specific topic would you like to learn more about?"
    
    elif intent == 'calculation' and tickers:
        ticker = tickers[0]
        stock = yf.Ticker(ticker)
        info = stock.info
        
        st.subheader(f"{ticker} Financial Metrics")
        
        metrics = {}
        
        if 'trailingPE' in info:
            metrics['P/E Ratio'] = info['trailingPE']
        if 'priceToBook' in info:
            metrics['Price to Book'] = info['priceToBook']
        if 'dividendYield' in info and info['dividendYield'] is not None:
            metrics['Dividend Yield'] = f"{info['dividendYield'] * 100:.2f}%"
        if 'returnOnEquity' in info and info['returnOnEquity'] is not None:
            metrics['Return on Equity'] = f"{info['returnOnEquity'] * 100:.2f}%"
        if 'debtToEquity' in info and info['debtToEquity'] is not None:
            metrics['Debt to Equity'] = info['debtToEquity']
        
        
        if metrics:
            metrics_df = pd.DataFrame({
                'Metric': metrics.keys(),
                'Value': metrics.values()
            })
            st.table(metrics_df)
            return f"Here are the key financial metrics for {ticker}."
        else:
            return f"Sorry, I couldn't retrieve financial metrics for {ticker}."
    
    else:
        
        return """
        I'm your stock market assistant. Here's how I can help you:
        
        - Get real-time stock prices (e.g., "What's the current price of AAPL?")
        - Show historical performance (e.g., "Show me MSFT's history for the past year")
        - Make price predictions (e.g., "Predict TSLA stock for the next two weeks")
        - Compare stocks (e.g., "Compare AAPL, MSFT, and GOOGL")
        - Provide recommendations (e.g., "What are some good dividend stocks?")
        - Offer learning resources (e.g., "How do I read stock charts?")
        - Calculate financial metrics (e.g., "What's the P/E ratio for AMZN?")
        
        What would you like to know about?
        """


if 'messages' not in st.session_state:
    st.session_state.messages = []


if page == "Chat":
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    
    if prompt := st.chat_input("Ask about stocks..."):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        
        with st.chat_message("assistant"):
            response = process_query(prompt)
            st.markdown(response)
        
        
        st.session_state.messages.append({"role": "assistant", "content": response})


elif page == "Market Overview":
    st.header("Market Overview")
    
    
    indices = {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "NASDAQ": "^IXIC",
        "Russell 2000": "^RUT"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Major Indices")
        for name, ticker in indices.items():
            price, change = get_stock_price(ticker)
            if price:
                if change >= 0:
                    st.markdown(f"**{name}**: {price:.2f} (🟢 +{change:.2f}%)")
                else:
                    st.markdown(f"**{name}**: {price:.2f} (🔴 {change:.2f}%)")
    
    with col2:
        st.subheader("Top Gainers/Losers")

        st.markdown("**Top Gainers**")
        st.markdown("- DEMO1: $45.67 (🟢 +8.2%)")
        st.markdown("- DEMO2: $32.45 (🟢 +6.7%)")
        st.markdown("- DEMO3: $78.90 (🟢 +5.3%)")
        
        st.markdown("**Top Losers**")
        st.markdown("- DEMO4: $12.34 (🔴 -7.8%)")
        st.markdown("- DEMO5: $56.78 (🔴 -5.6%)")
        st.markdown("- DEMO6: $23.45 (🔴 -4.2%)")
    
    
    st.subheader("Sector Performance")
    
    
    sectors = {
        "Technology": 2.5,
        "Healthcare": 1.8,
        "Financials": -0.5,
        "Consumer Cyclical": 0.7,
        "Industrials": -0.2,
        "Energy": 3.1,
        "Utilities": -1.2,
        "Real Estate": 0.3,
        "Materials": 1.0,
        "Communication Services": -0.8
    }
    
    
    sector_df = pd.DataFrame({
        "Sector": sectors.keys(),
        "Change (%)": sectors.values()
    })
    sector_df = sector_df.sort_values("Change (%)", ascending=False)
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sector_df["Sector"], sector_df["Change (%)"])
    
    
    for i, bar in enumerate(bars):
        if sector_df["Change (%)"].iloc[i] >= 0:
            bar.set_color("green")
        else:
            bar.set_color("red")
    
    plt.xticks(rotation=45, ha="right")
    plt.title("Sector Performance Today")
    plt.tight_layout()
    
    st.pyplot(fig)


elif page == "Learning Center":
    st.header("Stock Market Learning Center")
    
    
    tab1, tab2, tab3, tab4 = st.tabs(["Basics", "Analysis", "Strategies", "Practice"])
    
    with tab1:
        st.subheader("Understanding the Basics")
        st.markdown("""
        What is the Stock Market?
        The stock market is a collection of markets where stocks (pieces of ownership in businesses) are bought and sold. It refers to the exchanges where stocks and other securities are traded.
        
        ### Key Concepts:
        - **Stocks**: Represent ownership in a company
        - **Dividends**: Payments made by companies to shareholders
        - **Market Cap**: The total value of a company's outstanding shares
        - **Bull vs Bear Markets**: Rising vs falling market conditions
        - **Volatility**: The rate at which prices increase or decrease
        
        ### Common Stock Market Terms:
        - **IPO**: Initial Public Offering - when a company first offers shares to the public
        - **Broker**: An intermediary who buys and sells stocks for clients
        - **Portfolio**: Collection of investments owned by an individual
        - **Index**: A measurement of a section of the stock market
        """)
        
        st.video("https://www.youtube.com/watch?v=p7HKvqRI_Bo")
    
    with tab2:
        st.subheader("Stock Analysis Methods")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Fundamental Analysis
            Evaluating a stock based on the company's financial health and business model.
            
            **Key metrics include:**
            - P/E Ratio
            - EPS (Earnings Per Share)
            - P/B Ratio (Price to Book)
            - Debt-to-Equity Ratio
            - Revenue Growth
            - Profit Margins
            """)
        
        with col2:
            st.markdown("""
            ### Technical Analysis
            Analyzing statistical trends from trading activity to identify patterns.
            
            **Common techniques include:**
            - Moving Averages
            - Support and Resistance Levels
            - RSI (Relative Strength Index)
            - MACD (Moving Average Convergence Divergence)
            - Bollinger Bands
            - Volume Analysis
            """)
        
        st.markdown("### Example Chart Patterns")
        st.image("https://www.investopedia.com/thmb/vfYw4S-DM44vs9T5CAKjX67J1g4=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/dotdash_Final_Technical_Analysis_Chart_Patterns_Sep_2020-01-38ae17a591e443978f60b8410e8e274f.jpg", 
                 caption="Common chart patterns used in technical analysis")
    
    with tab3:
        st.subheader("Investment Strategies")
        
        st.markdown("""
        ### Value Investing
        Buying stocks that appear to be undervalued based on fundamental analysis.
        
        ### Growth Investing
        Investing in companies expected to grow at an above-average rate compared to the market.
        
        ### Income Investing
        Focusing on stocks that generate regular income through dividends.
        
        ### Dollar-Cost Averaging
        Investing a fixed amount regularly, regardless of share price.
        
        ### Diversification
        Spreading investments across various assets to reduce risk.
        """)
        
        st.info("Remember: Each strategy has its own risk profile and time horizon. It's important to choose a strategy that aligns with your financial goals and risk tolerance.")
        
        # Simple strategy comparison
        strategies = {
            "Strategy": ["Value", "Growth", "Income", "Index"],
            "Risk Level": ["Medium", "High", "Low", "Medium-Low"],
            "Time Horizon": ["Long", "Medium-Long", "Medium", "Long"],
            "Focus": ["Undervalued Companies", "Companies with High Growth Potential", "Dividend-Paying Companies", "Market-Wide Exposure"]
        }
        
        st.table(pd.DataFrame(strategies))
    
    with tab4:
        st.subheader("Practice Resources")
        
        st.markdown("""
        ### Paper Trading
        Practice trading with virtual money before investing real funds.
        
        ### Market Simulators
        Try these virtual trading platforms:
        - **MarketWatch Virtual Stock Exchange**
        - **Investopedia Stock Simulator**
        - **TD Ameritrade paperMoney**
        
        ### Case Studies
        """)
        
        # Sample case study
        with st.expander("Case Study: Evaluating a Tech Stock"):
            st.markdown("""
            ### Company: TechCorp (Fictional)
            
            **Background:**
            - Founded in 2010
            - Creates enterprise software solutions
            - Recent expansion into cloud services
            
            **Financial Metrics:**
            - Revenue: $500M (20% YoY growth)
            - P/E Ratio: 25
            - Profit Margin: 15%
            - Debt-to-Equity: 0.3
            
            **Questions to Consider:**
            1. Is this stock overvalued or undervalued compared to the industry average?
            2. How sustainable is the company's growth rate?
            3. What are the key risks for this company?
            4. Would this fit better in a growth or value portfolio?
            """)
        
        st.markdown("### Practice Exercises")
        
        exercise = st.selectbox("Select an exercise to practice:", 
                                ["Basic Stock Valuation", 
                                 "Technical Indicator Analysis", 
                                 "Portfolio Allocation", 
                                 "Risk Assessment"])
        
        if exercise == "Basic Stock Valuation":
            st.markdown("""
            **Exercise: Calculate the following for a stock with these metrics:**
            - Current Price: $50
            - Earnings Per Share (EPS): $2.50
            - Book Value Per Share: $20
            - Annual Dividend: $1.00
            
            1. P/E Ratio
            2. P/B Ratio
            3. Dividend Yield
            4. PEG Ratio (if growth rate is 15%)
            
            [Click for Solution]
            """)


if __name__ == "__main__":
    pass  