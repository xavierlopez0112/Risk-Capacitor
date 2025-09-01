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
import time
import feedparser
import ssl
import urllib.request
from dotenv import load_dotenv
from pandas import DataFrame
from datetime import datetime, timedelta
from yahooquery import search, Ticker, Screener
from sklearn.metrics import mean_squared_error, r2_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from urllib.parse import quote_plus
from transformers import pipeline
load_dotenv()


pipe = pipeline("text-classification", model="ProsusAI/finbert")
print(pipe('Stocks rallied and the British pound gained.'))

s = Screener()
openai.api_key=os.getenv("OPENAIAPI_KEY")
newsapi_key=os.getenv("NEWSAPI_KEY")
newsdata_api_key = os.getenv("NEWSDATA_API_KEY")
finnhub_api_key=os.getenv("FINNHUB_API_KEY")
NEWSDATA_BASE_URL = "https://newsdata.io/api/1/news"
NEWS_BASE_URL = "https://newsapi.org/v2/everything"
analyzer = SentimentIntensityAnalyzer()
SENTIMENT_THRESHOLD=0.1
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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

        rss_urls = [f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
                    f"https://news.google.com/rss/search?q={quote_plus(symbol)}+yahoo+finance&hl=en-US&gl=US&ceid=US:en",
                    f"https://feeds.marketwatch.com/marketwatch/companyreport?symbol={symbol}",
                    ]
        for rss_url in rss_urls:
            try:
                try:
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE

                    req = urllib.request.Request(
                        rss_url,
                        headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    )
                    with urllib.request.urlopen(req, timeout=10, context = ssl_context) as response:
                        feed_content = response.read()
                    feed = feedparser.parse(feed_content)
                except:
                    feed = feedparser.parse(rss_url)
                print(f"  RSS Status: {feed.get('status', 'unknown')}")
                print(f"  Found {len(feed.entries)} Yahoo RSS entries from this URL")
                print(f"Found {len(feed.entries)} Yahoo RSS entries for {symbol}")

                for entry in feed.entries[:max_articles]:
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date=datetime(*entry.published_parsed[:6])
                        days_old =(datetime.now() - pub_date).days
                        if days_old > 7:
                            continue
                    title = entry.get('title','')
                    summary = entry.get('summary','')

                    text_to_check=f"{title} {summary}".lower()
                    symbol_lower = symbol.lower()
                    company_words = company_name.lower().split()[:2]
                    is_relevant = (
                        symbol_lower in text_to_check
                        or any(word in text_to_check for word in company_name if len(word) > 3)
                        or any(k in text_to_check for k in [
                            'earnings', 'stock', 'shares','revenue','profit',
                             'quarterly','results','financial','guidance'
                        ])
                    )
                    if is_relevant:
                           articles.append({
                                'title': title,
                                'summary': summary,
                                'published': entry.get('published',''),
                                'link': entry.get('link',''),
                                'source': 'yahoo rss'
                            })
            except Exception as e:
                print(f"Error parsing RSS for {symbol}: {e}")


        return articles
    except Exception as e:
        print(f"Error getting Yahoo News for{symbol}: {e}")
                


def get_google_rss_news(symbol,company_name, max_articles = 20):
    try:
        articles = []
        encoded_symbol = quote_plus(symbol)
        encoded_company = quote_plus(company_name)
        rss_urls = [
            f"https://news.google.com/rss/search?q={encoded_symbol}+stock&hl=en-US&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={encoded_company}+earnings&hl=en-US&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q=\"{encoded_symbol}\"+finance&hl=en-US&gl=US&ceid=US:en"
        ]
        for rss_url in rss_urls:
            try:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                req = urllib.request.Request(
                    rss_url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                
                with urllib.request.urlopen(req, timeout=10, context = ssl_context) as response:
                    feed_content = response.read()
                    
                feed = feedparser.parse(feed_content)
                print(f"  RSS Status: {feed.get('status', 'unknown')}")
                print(f"  Found {len(feed.entries)} RSS entries from this URL")
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
                    symbol_lower = symbol.lower()
                    company_words = company_name.lower().split()[:2]
                    is_relevant = (
                        symbol_lower in text_to_check
                        or any(word in text_to_check for word in company_name if len(word) > 3)
                        or any(k in text_to_check for k in [
                            'earnings', 'stock', 'shares','revenue','profit',
                             'quarterly','results','financial','guidance'
                        ])
                    )
                    if is_relevant:
                           articles.append({
                                'title': title,
                                'summary': summary,
                                'published': entry.get('published',''),
                                'link': entry.get('link',''),
                                'source': 'Google RSS'
                            })
            except Exception as e:
                print(f"Error parsing RSS for {symbol}: {e}")


        return articles
    except Exception as e:
        print(f"Error getting Google News for{symbol}: {e}")

def get_finnhub_news(symbol, company_name, max_articles=20):
    """Get news from Finnhub API"""
    try:
        if not finnhub_api_key:
            print("Finnhub API key not found in environment variables")
            return []
        
        articles = []

        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        url = f"https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': symbol,
            'from': from_date,
            'to': to_date,
            'token': finnhub_api_key
        }
        
        print(f"Fetching Finnhub news for {symbol} from {from_date} to {to_date}")
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            news_data = response.json()
            print(f"Found {len(news_data)} Finnhub articles for {symbol}")
            
            for item in news_data[:max_articles]:

                title = item.get('headline', '')
                summary = item.get('summary', '')
                
                timestamp = item.get('datetime', 0)
                if timestamp:
                    pub_date = datetime.fromtimestamp(timestamp)
                    published = pub_date.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    published = ''
                
                text_to_check = f"{title} {summary}".lower()
                symbol_lower = symbol.lower()
                
                is_relevant = (
                    symbol_lower in text_to_check
                    or any(word in text_to_check for word in company_name.lower().split() if len(word) > 3)
                    or any(keyword in text_to_check for keyword in [
                        'earnings', 'stock', 'shares', 'revenue', 'profit',
                        'quarterly', 'results', 'financial', 'guidance'
                    ])
                )
                
                if is_relevant and title.strip():
                    articles.append({
                        'title': title,
                        'summary': summary,
                        'published': published,
                        'link': item.get('url', ''),
                        'source': 'finnhub'
                    })
        
        elif response.status_code == 429:
            print(f"Finnhub API rate limit exceeded for {symbol}")
        else:
            print(f"Finnhub API error {response.status_code} for {symbol}")
            
        print(f"Total relevant Finnhub articles for {symbol}: {len(articles)}")
        return articles
        
    except Exception as e:
        print(f"Error getting Finnhub news for {symbol}: {e}")
        return []




def get_finbert_sentiment(text):
    """Analyze sentiment using finBERT pipeline"""
    try:
        if not text or not text.strip():
            return 0.0, "NEUTRAL"
        text = text[:512]
        result = pipe(text)

        label = result[0]['label'].upper()
        score = result[0]['score']

        if label == 'POSITIVE':
            sentiment_score = score
        elif label == 'NEGATIVE':
            sentiment_score =-score
        else:
            sentiment_score =0.0
        return sentiment_score,label
    except Exception as e:
        print(f"Error analyzing sentiment with finBERT: {e}")
        return 0.0, "NEUTRAL" 







def get_stock_sentiment(symbol, company_name):

    """Get sentiment from news for a stock"""
    try:
        print(f"Getting Yahoo RSS news headlines for{symbol}")
        all_articles = []

        yahoo_articles = get_yahoo_rss_news(symbol,company_name)
        all_articles.extend(yahoo_articles)
        print(f"Yahoo Articles: {len(yahoo_articles)}")

        google_articles = get_google_rss_news(symbol,company_name)
        all_articles.extend(google_articles)
        print(f"Google Articles: {len(google_articles)}")

        finnhub_articles = get_finnhub_news(symbol,company_name)
        all_articles.extend(finnhub_articles)
        print(f"Finnhub Articles: {len(finnhub_articles)}")

        if not all_articles:
            print(f"No articles found for {symbol} / {company_name}")
            return 0.0, "NEUTRAL", 0
        unique_articles =[]
        seen_titles=set()

        for article in all_articles:
            title_lower= article.get('title','').lower().strip()
            if title_lower and title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_articles.append(article)
                
        sentiments = []
        for article in unique_articles:
            title = article.get('title','')
            summary = article.get('summary','')
            text = f"{title}.{summary}" if summary else title

            if text.strip():
                sentiment_score, sentiment_label = get_finbert_sentiment(text)
                sentiments.append(sentiment_score)
        

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
                optimal_model = poly
                optimal_degree = degree

        if optimal_model is None:
            print(f"Could not find fit for {symbol}")
            return None
            
        coeffs = optimal_model.c
        current_price = closing_prices[-1]
        predicted_30_day = np.polyval(coeffs, len(closing_prices)+30)
        pct_change = ((predicted_30_day - current_price) / current_price) *100
        terms = []
        for i, c in enumerate(coeffs):
            pwr = len(coeffs)-1-i
            if pwr == 0: terms.append(f"{c:.6g}")
            elif pwr == 1: terms.append(f"{c:.6g}x")
            else: terms.append(f"{c:.6g}x^{pwr}")
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
    test_stocks = all_stocks[:5]
    for i, symbol in enumerate(test_stocks,1):
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

        




def test_single_stock():
    """Test function to debug a single stock"""
    test_symbol = "AAPL"  # Use a major stock that should have news
    print(f"Testing sentiment analysis for {test_symbol}...")
    
    try:
        ticker = Ticker(test_symbol)
        company_name = ticker.quote_type[test_symbol].get("longName", test_symbol)
        print(f"Company name: {company_name}")
        
        sentiment_score, sentiment_label, num_articles = get_stock_sentiment(test_symbol, company_name)
        print(f"Result: {sentiment_label} (score: {sentiment_score:.3f}, articles: {num_articles})")
        
    except Exception as e:
        print(f"Error in test: {e}")
    











def main():
    while True:
        print("1 For Manually Checking Stocks")
        print("2 For Automatic Top Stocks")
        print("3")
        print("Q to Quit")
        choice = input("Choose Option 1 or 2 ").strip()
        if choice == "1":
            manual_analysis()   
        elif choice == "2":
            auto_sentiment_analysis()
        elif choice =="3":
            test_single_stock()
        elif choice.lower() =="q":
            break
        else:
            print("Invalid choice")
if __name__ == "__main__":
    main() 
