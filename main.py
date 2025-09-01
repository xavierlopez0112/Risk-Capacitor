import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import scipy as sci
import pandas as pd
import os
import openai
import requests
import torch
import feedparser
from dotenv import load_dotenv
from pandas import DataFrame
from datetime import datetime, timedelta
from yahooquery import search, Ticker, Screener
from sklearn.metrics import mean_squared_error, r2_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
load_dotenv()


pipe = pipeline("text-classification", model="ProsusAI/finbert")


s = Screener()
openai.api_key=os.getenv("OPENAIAPI_KEY")
newsapi_key=os.getenv("NEWSAPI_KEY")
newsdata_api_key = os.getenv("NEWSDATA_API_KEY")
NEWSDATA_BASE_URL = "https://newsdata.io/api/1/news"
NEWS_BASE_URL = "https://newsapi.org/v2/everything"
analyzer = SentimentIntensityAnalyzer()
SENTIMENT_THRESHOLD=0.1

most_actives = s.get_screeners(["most_actives"])['most_actives']['quotes']
undervalued = s.get_screeners(["undervalued_large_caps"])['undervalued_large_caps']['quotes']
day_gainers = s.get_screeners(["day_gainers"])['day_gainers']['quotes']
day_losers = s.get_screeners(["day_losers"])['day_losers']['quotes']
most_shorted = s.get_screeners(["most_shorted_stocks"])['most_shorted_stocks']['quotes']
growth_tech = s.get_screeners(["growth_technology_stocks"])['growth_technology_stocks']['quotes']

most_active_symbols = [stock['symbol'] for stock in most_actives[:10]]
undervalued_symbols = [stock['symbol'] for stock in undervalued[:5]]
day_gainer_symbols = [stock['symbol'] for stock in day_gainers[:10]]
day_loser_symbols = [stock['symbol'] for stock in day_losers[:5]]
most_shorted_symbols = [stock['symbol'] for stock in most_shorted[:5]]
growth_tech_symbols = [stock['symbol'] for stock in growth_tech[:5]]

all_stocks = most_active_symbols + undervalued_symbols + day_gainer_symbols + day_loser_symbols + most_shorted_symbols + growth_tech_symbols
def get_yahoo_rss_news(symbol, company_name, max_articles=20):
    """Get news from Yahoo Finance RSS feed"""
    try:
        articles = []
        RSS_URL = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        try:
            feed = feedparser.parse(RSS_URL)
            print(f"Found {len(feed.entries)} RSS entries for {symbol}")

            for entry in feed.entries[:max_articles]:
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date=datetime(*entry.published_parsed[:6])
                    days_old =(datetime.now() - pub_date).days
                    if days_old > 7:
                        continue
                title = entry.get('title','')
                summary = entry.get('summary','')

                text_to_check=f"{title} {summary}".lower()
                if(symbol.lower() in text_to_check or 
                    company_name.lower() in text_to_check or 
                    any(keyword in text_to_check for keyword in ['earnings', 'stock', 'shares','revenue','profit'])):

                        articles.append({
                            'title': title,
                            'summary': summary,
                            'published': entry.get('published',''),
                            'link': entry.get('link','')
                        })
        except Exception as e:
            print(f"Error parsing RSS for {symbol}: {e}")


        return articles
    except Exception as e:
        print(f"Error getting Yahoo News for{symbol}: {e}")
                
def get_finbert_sentiment(text)
    """Analyze sentiment using finBERT pipeline"""
    try:
        if not text in text.strip()
            return 0.0, "NEUTRAL"
        text = text[:512]
        result = pipe(text)









def get_stock_sentiment(symbol, company_name):

    """Get sentiment from news for a stock"""
    try:
        to_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        search_queries = [
            f'"{symbol}"',
            f'"{company_name}"',
            f'{symbol} stock',
            f'{company_name} earnings'
        ]
        all_articles=[]
        for query in search_queries:

            params = {
                'apikey' : newsdata_api_key,
                'q':f'"{symbol}" OR "{company_name}"',
                'language': 'en',
                'country': 'us',
                'size': 10,
                'category': 'business',
                'to_date': to_date,
                'from_date': from_date
                
            }
            try:
                response = requests.get(NEWSDATA_BASE_URL, params=params, timeout=10)

                if response.status_code !=200:
                    print(f"NewsData API error {response.status_code} for query: {query}")
                    continue
                
                news_data = response.json()

                if news_data.get('status') != 'success':
                    print(f"NewsData API error for query: {query}: {news_data}")
                    continue
                
                articles = news_data.get('results', [])
                all_articles.extend(articles)

                import time
                time.sleep(0.1)
            except Exception as e:
                return 0.0, "NEUTRAL", 0

        if not articles:
            print(f"No articles found for {symbol} / {company_name}")
            return 0.0, "NEUTRAL", 0
        sentiments =[]

        for article in articles:
            title = article.get('title', '')
            description = article.get('description','')
            content = article.get('content','')

            text = f"{title}  {description}" if description else f"{title} {content[:200]}"
            if text.strip():
                scores = analyzer.polarity_scores(text)
                sentiments.append(scores['compound'])
        num_articles = len(sentiments)
        if num_articles > 0:
            avg_sentiment = sum(sentiments) / num_articles
        else: 
            avg_sentiment = 0.0

        if avg_sentiment > SENTIMENT_THRESHOLD:
            sentiment_label = "POSITIVE"
        elif avg_sentiment < -SENTIMENT_THRESHOLD:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "NEUTRAL"

        return avg_sentiment, sentiment_label, num_articles
    except Exception as e:
        print(f"Error getting Sentiment for {symbol}: {e}")
        return 0.0, "NEUTRAL", 0


def analyze_stock(symbol, user_prompt=None, company_name=None):
    try:
        closing_prices = None
        trading_days = None
        optimal_model = None
        optimal_degree = None
        lowest_mse = float('inf')
        if not company_name:
            ticker = Ticker(symbol)
            company_name = ticker.quote_type[symbol].get("longName", symbol)
        print(f"You Selected {user_prompt} / {symbol}")

        end_date = datetime.today()
        start_date = end_date - timedelta(days=90)
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)

        if data is None or data.empty:
            print(f"No data found for {ticker}")
            return None
        print(f"Getting sentiment for {symbol}...")
        sentiment_score, sentiment_label, num_articles = get_stock_sentiment(symbol, company_name)

        closing_prices = data['Close'].dropna().values
        if len(closing_prices) <10:
            print("Not Enough Data for Closing Prices")
            return None
        

        trading_days = np.arange(len(closing_prices))

        for degree in range (1,5):
            coefficients = np.polyfit(trading_days, closing_prices, degree).flatten()
            poly = np.poly1d(coefficients)
            preds = poly(trading_days)  
            mse = mean_squared_error(closing_prices, preds)

            if mse < lowest_mse:
                lowest_mse = mse
                optimal_model=poly
                optimal_degree=degree
            if optimal_model is None:
                print(f"Could not find fit for {symbol}")
                return None
            coeffs = optimal_model.c
        current_price = closing_prices[-1]
        predicted_30_day = np.polyval(coeffs, len(closing_prices)+30)
        pct_change = ((predicted_30_day - current_price) / current_price) *100
        terms = []
        for i, c in enumerate(coeffs):
            pow = len(coeffs)-1-i
            if pow == 0: terms.append(f"{c:.6g}")
            elif pow == 1: terms.append(f"{c:.6g}x")
            else: terms.append(f"{c:.6g}x^{pow}")
        equation = " + ".join(terms).replace("+ -", "-")
        print(f"{company_name}'s Equation for Best Fit:\n {equation} \n MSE is {lowest_mse:0.3f}")

        x = sym.symbols('x')
        poly_expr = sum(sym.Float(c) * x**i for i, c in enumerate(coeffs[::-1]))
        derivative = sym.diff(poly_expr, x)
        deriv_values = [derivative.subs(x,day) for day in trading_days]
        critical_points = sym.solve(sym.Eq(derivative, 0) ,x)
        real_crit = [float(p) for p in critical_points if p.is_real and 0<=p and p<=len(closing_prices)-1]





# EDIT LATER
#        plt.figure(figsize = (12,6))
#        plt.subplot(2,1,1)
#        plt.plot(trading_days, closing_prices, label = "Actual Prices", linewidth=2)
#        plt.plot(trading_days, optimal_model(trading_days),label =f"Degree: {optimal_degree}")
        plt.xlabel("Trading Days")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)


        plt.subplot(2,1,2)
        plt.plot(trading_days, deriv_values, color='green', label = "Derivative")
        plt.axhline(0, color='red', linestyle=':')
        plt.title("Derivative of Polynomial Fit")

        for cr in real_crit:
            plt.axvline(cr, color ='red', alpha=0.6)
            plt.scatter(cr, optimal_model(cr), color ='red')
        plt.legend()
        plt.grid(True)
        return {
            'symbol': symbol,
            'company_name': company_name,
            'mse': mse,
            'coeffs': coeffs,
            'current_price': current_price,
            'predicted_30_day': predicted_30_day,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'num_articles':num_articles,
            }
    except Exception as e:
        print(f"Error Analyzing{company_name}: {e}")
        return None


def manual_analysis():
    while True:
        user_prompt = input("Enter Stock Ticker / Stock Name ")
        
        if user_prompt.lower() == "q":
            break

        if user_prompt == "":
            user_prompt = "S&P 500"
            symbol = "^GSPC"
        else:
            result = search(user_prompt)
            user_prompt = user_prompt.upper()
     
            if result and "quotes" in result and len(result["quotes"]) > 0:
                symbol = result["quotes"][0]["symbol"]
            else: 
                symbol = user_prompt
        result = analyze_stock(symbol,user_prompt)
        if result:
            print("Analysis Complete")
        

def auto_sentiment_analysis():
    analysis_results=[]
    
    for i, symbol in enumerate(all_stocks,1):
        print(f"Processing {symbol}")
        try:
            result = analyze_stock(symbol)
            if result:
                analysis_results.append(result)
                print(f"Successfully analyzed {symbol}")
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")  # Debug line
            continue
    
    print(f"Total analysis results: {len(analysis_results)}")  # Debug line
    if analysis_results:
        print("DEBUG: analysis_results =", analysis_results)
        sentiment_counts={}
        sentiment_scores = [r.get('sentiment_score', 0) for r in analysis_results if 'sentiment_score' in r]
        print("DEBUG: sentiment_scores =", sentiment_scores)
        if sentiment_scores:
            print(f"Sentiment scores range: {min(sentiment_scores):.3f} to {max(sentiment_scores):.3f}")
            print(f"Average sentiment score: {sum(sentiment_scores)/len(sentiment_scores):.3f}")
        else:
            print("DEBUG: No sentiment scores found")
    else:
        print("DEBUG: analysis_results is empty or None")
        for result in analysis_results:
            label = result['sentiment_label']
            sentiment_counts[label] = sentiment_counts.get(label,0) + 1
        print("Sentiment Distribution:")
        for label, count in sentiment_counts.items():
            print(f" {label}: {count}")
        print("\nFirst few results:")
        for result in analysis_results[:3]:
            print(f"  {result['symbol']}: {result['sentiment_label']} (score: {result.get('sentiment_score', 'N/A')})")
    positive_stocks = [r for r in analysis_results if r['sentiment_label'] == 'POSITIVE']
    print(f"Positive stocks found: {len(positive_stocks)}")
    if not positive_stocks:
        print("No positive sentiment candidates found.")
        print("Showing all candidates for debugging:")
        all_candidates = sorted(analysis_results, key=lambda x: x['mse'])
    best_candidates = sorted(positive_stocks, key=lambda x: x['mse'])
    print(f"Best candidates: {len(best_candidates)}")

    if not best_candidates:
        print("No positive sentiment candidates found.")
        return
    
    for rank, result in enumerate(best_candidates[:15],1):
        current = result['current_price']
        predicted = result['predicted_30_day']
        pct_change = ((predicted - current) / current) * 100

        print(f"{rank:<4} {result['symbol']} {result['company_name']} "
              f"MSE: {result['mse']} Sentiment: {result['sentiment_score']} "
              f"${result['current_price']:2f} → ${result['predicted_30_day']:2f} ({pct_change:+.1f}%)")

        




    











def main():
    while True:
        print("1 For Manually Checking Stocks")
        print("2 For Automatic Top Stocks")
        print("Q to Quit")
        choice = input("Choose Option 1 or 2 ").strip()
        if choice == "1":
            manual_analysis()   
        elif choice == "2":
            auto_sentiment_analysis()
        elif choice.lower() =="q":
            break
        else:
            print("Invalid choice")
if __name__ == "__main__":
    main()
