import feedparser
import requests
from yahooquery import search, Ticker, Screener
from transformers import pipeline
import ssl
import urllib.request
from urllib.parse import quote_plus
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv



SENTIMENT_THRESHOLD=0.1
newsapi_key=os.getenv("NEWSAPI_KEY")
newsdata_api_key = os.getenv("NEWSDATA_API_KEY")
finnhub_api_key=os.getenv("FINNHUB_API_KEY")
load_dotenv()






pipe = pipeline("text-classification", model="ProsusAI/finbert")
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