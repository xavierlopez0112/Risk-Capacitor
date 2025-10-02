import yfinance as yf
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import scipy as sci
import pandas as pd
import os
import sys
import openai
import requests
import traceback
import warnings
import time
import feedparser
import ssl
import urllib.request
from dotenv import load_dotenv
from pandas import DataFrame
from datetime import datetime, timedelta
from yahooquery import search, Ticker, Screener
from sklearn.metrics import mean_squared_error, r2_score
from urllib.parse import quote_plus
from transformers import pipeline
from scipy.interpolate import CubicSpline, BSpline, splrep, splev
from sklearn.model_selection import KFold
from scipy import stats
from arch import arch_model
load_dotenv()
warnings.filterwarnings('ignore')




pipe = pipeline("text-classification", model="ProsusAI/finbert")


s = Screener()
openai.api_key=os.getenv("OPENAIAPI_KEY")
newsapi_key=os.getenv("NEWSAPI_KEY")
newsdata_api_key = os.getenv("NEWSDATA_API_KEY")
finnhub_api_key=os.getenv("FINNHUB_API_KEY")
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

def calculate_log_returns(prices):
    return np.log(prices[1:] / prices[:-1])

def estimate_gbm_params(prices):
    log_returns = calculate_log_returns(prices)

    mu_daily = np.mean(log_returns)
    sigma_daily = np.std(log_returns, ddof=1)

    mu_annual = mu_daily*252
    sigma_annual = sigma_daily * np.sqrt(252)
    return {
        'mu_daily': mu_daily,
        'sigma_daily': sigma_daily,
        'mu_annual': mu_annual,
        'sigma_annual': sigma_annual,
        'log_returns': log_returns
    }
def gbm_simulation(S0, mu, sigma, T, dt, n_simulations=1000):
    """
    S0: Initial Stock Price
    mu: Drift (daily)
    sigma: Volatility (daily)
    T: Time Horizon (in days) - How far it is simulating
    dt: Time Step (1 day = 1.0) - How often price is computed
    """
    n_steps = int(T / dt)
    simulations = np.zeros((n_simulations, n_steps+1))
    simulations[:, 0] = S0

    random_shocks = np.random.normal(0,1, (n_simulations, n_steps))

    for t in range(1, n_steps + 1):
        drift = (mu - 0.5*sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * random_shocks[:, t-1]
        simulations[:, t] = simulations[:, t-1] * np.exp(drift + diffusion)
    return simulations

def garch_volatility_forecast(log_returns, forecast_days=30):

    try:
        returns_pct = log_returns * 100

        model = arch_model(
            returns_pct,
            vol='Garch',
            p=1,
            q=1,
            mean='Zero',
            dist='normal'
        )
        fitted_model = model.fit(disp='off', show_warning=False)
        omega = fitted_model.params['omega']
        alpha = fitted_model.params['alpha[1]']
        beta = fitted_model.params['beta[1]']

        cond_var = fitted_model.conditional_volatility ** 2

        forecast = fitted_model.forecast(horizon = forecast_days)
        forecasted_vars = forecast.variance.values[-1, :]

        forecasted_vol = np.sqrt(forecasted_vars) / 100
        cond_var_original = cond_var ** 2 / 10000

        print(f"\nGARCH(1,1) Parameters (arch package):")
        print(f"  ω (omega): {omega:.6f}")
        print(f"  α (alpha): {alpha:.6f}")
        print(f"  β (beta): {beta:.6f}")
        print(f"  Persistence (α+β): {alpha + beta:.6f}")
        print(f"  AIC: {fitted_model.aic:.2f}")
        print(f"  BIC: {fitted_model.bic:.2f}")

        return {
            'conditional_variance': cond_var_original,
            'forecasted_volatility': forecasted_vol,
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'avg_forecasted_vol': np.mean(forecasted_vol),
            'fitted_model': fitted_model,
            'method': 'arch_package'
        }
    except Exception as e:
        print(f"Garch estimation with ARCH failed: {e}")

        try:
            returns = log_returns - np.mean(log_returns)
            omega = np.var(returns) * 0.1
            alpha = 0.1
            beta = 0.8
            if alpha + beta >= 1:
                alpha = 0.05
                beta = 0.9
                
            n = len(returns)
            cond_var = np.zeros(n)
            cond_var[0] = np.var(returns)
            
            for t in range(1, n):
                cond_var[t] = omega + alpha * returns[t-1]**2 + beta * cond_var[t-1]
            
            # Forecast volatility
            last_return = returns[-1]
            last_var = cond_var[-1]
            
            forecasted_vars = np.zeros(forecast_days)
            forecasted_vars[0] = omega + alpha * last_return**2 + beta * last_var
            
            # Multi-step ahead forecast
            for t in range(1, forecast_days):
                forecasted_vars[t] = omega + (alpha + beta) * forecasted_vars[t-1]
            
            forecasted_vol = np.sqrt(forecasted_vars)
            
            return {
                'conditional_variances': cond_var,
                'forecasted_volatility': forecasted_vol,
                'omega': omega,
                'alpha': alpha,
                'beta': beta,
                'avg_forecast_vol': np.mean(forecasted_vol)
            }
            
        except Exception as e:
            print(f"GARCH estimation failed: {e}")
            # Fallback to constant volatility
            constant_vol = np.std(log_returns)
            return {
                'conditional_variances': None,
                'forecasted_volatility': np.full(forecast_days, constant_vol),
                'omega': None,
                'alpha': None,
                'beta': None,
                'avg_forecast_vol': constant_vol
            }
def garch_volatility_forecast_enhanced(log_returns, forecast_days = 30, p=1, q=1, dist='normal'):
    try:
        returns_pct = log_returns * 100
        model = arch_model(
            returns_pct,
            vol='Garch',
            p=p,
            q=q,
            mean='Zero',
            dist=dist
        )

        fitted_model = model.fit(disp='off', show_warning=False)
        forecast = fitted_model.forecast(horizon=forecast_days)
        forecasted_vars = forecast.variance.values[-1, :]
        forecasted_vol = np.sqrt(forecasted_vars) / 100

        params_dict = {
            'omega': fitted_model.params['omega'] 
        }
        for i in range(1, p + 1):
            params_dict[f'alpha[{i}]'] = fitted_model.params[f'alpha[{i}]']
        
        for i in range(1, q + 1):
            params_dict[f'beta[{i}]'] = fitted_model.params[f'beta[{i}]']
        
        print(f"Garch({p},{q}) Parameters with {dist} distribution:")
        for param, value in params_dict.items():
            print(f"  {param}: {value:.6f}")
        print(f"AIC: {fitted_model.aic:.2f}")
        print(f"BIC: {fitted_model.bic:.2f}")
        
        return {
            'conditional_variances': fitted_model.conditional_volatility ** 2 / 10000,
            'forecasted_volatility': forecasted_vol,
            'parameters': params_dict,
            'avg_forecast_vol': np.mean(forecasted_vol),
            'fitted_model': fitted_model,
            'model_spec': f'GARCH({p},{q})',
            'distribution': dist,
            'method': 'arch_package_advanced'
        }
        
    except Exception as e:
        print(f"Advanced GARCH estimation failed: {e}")
        return garch_volatility_forecast(log_returns, forecast_days)
def compare_garch_models(log_returns, forecast_days=30):

    print("\n" + "="*70)
    print("GARCH MODEL COMPARISON")
    print("="*70)

    models_to_test = [
        {'p': 1, 'q': 1, 'dist': 'normal', 'name': 'GARCH(1,1) - Normal'},
        {'p': 1, 'q': 1, 'dist': 't', 'name': 'GARCH(1,1) - Student-t'},
        {'p': 1, 'q': 1, 'dist': 'skewt', 'name': 'GARCH(1,1) - Skewed-t'},
        {'p': 2, 'q': 1, 'dist': 'normal', 'name': 'GARCH(2,1) - Normal'},
        {'p': 1, 'q': 2, 'dist': 'normal', 'name': 'GARCH(1,2) - Normal'},
        {'p': 2, 'q': 1, 'dist': 't', 'name': 'GARCH(2,1) - Student-t'},
    ]
    
    results = []

    for spec in models_to_test:
        try:
            result = garch_volatility_forecast_enhanced(log_returns, forecast_days = forecast_days, p=spec['p'], q=spec['q'], dist=spec['dist'])
            if result['fitted_model'] is not None:
                results.append({
                    'name': spec['name'],
                    'p': spec['p'],
                    'q': spec['q'],
                    'dist': spec['dist'],
                    'aic': result['fitted_model'].aic,
                    'bic': result['fitted_model'].bic,
                    'log_likelihood': result['fitted_model'].loglikelihood,
                    'avg_forecast_vol': result['avg_forecast_vol'],
                    'result': result
                })
                print(f"Success - AIC: {result['fitted_model'].aic:.2f}, BIC: {result['fitted_model'].bic:.2f}")
            else:
                print(f"Failed - Model could not be estimated")
                
        except Exception as e:
            print(f"  ✗ Failed - {str(e)[:50]}")
            continue
    
    if not results:
        print("All advanced models failed. Using simplified GARCH.")
        return garch_volatility_forecast(log_returns, forecast_days)
    df = pd.DataFrame(results)
    df = df.sort_values('aic')

    best_by_aic = results[df.index[0]]
    best_by_bic = df.sort_values('bic').iloc[0]
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(f"Best by AIC: {best_by_aic['name']}")
    print(f"Best by BIC: {best_by_bic['name']}")
    return best_by_aic['result'], df


    


def stochastic_analysis(analysis_result, n_simulations=1000, forecast_days = 30, compare_models=False):
    try:
        closing_prices = analysis_result['closing_prices']
        symbol = analysis_result['symbol']
        current_price = analysis_result['current_price']

        print(f"\nRunning Stochastic Analysis for {symbol}....")

        gbm_params = estimate_gbm_params(closing_prices)
        print(f"GBM Parameters:")
        print(f"  Daily drift (μ): {gbm_params['mu_daily']:.4f}")
        print(f"  Daily volatility (σ): {gbm_params['sigma_daily']:.4f}")
        print(f"  Annual drift: {gbm_params['mu_annual']:.2%}")
        print(f"  Annual volatility: {gbm_params['sigma_annual']:.2%}")

        if compare_models:
            best_garch, comparison_df = compare_garch_models(gbm_params['log_returns'], forecast_days)
            garch_result = best_garch
        else:
            garch_result = garch_volatility_forecast(gbm_params['log_returns'], forecast_days)
            comparison_df= None
        has_valid_garch = False
        if isinstance(garch_result, dict):
            forecasted_vol = garch_result.get('forecasted_volatility')
            if forecasted_vol is not None:
                try:
                    vol_length = len(forecasted_vol)
                    has_valid_garch = vol_length > 0
                except:
                    has_valid_garch = False
        if has_valid_garch:
            simulations = np.zeros((n_simulations, forecast_days + 1))
            simulations[:, 0] = current_price

            for t in range(1, forecast_days + 1):
                vol = best_garch['forecasted_volatility'][t-1]
                random_shocks = np.random.normal(0,1,n_simulations)
                drift = (gbm_params['mu_daily'] - 0.5 * vol**2)
                diffusion = vol * random_shocks
                simulations[:, t] = simulations[:, t-1] * np.exp(drift + diffusion)
        else:
            simulations = gbm_simulation(
                S0 = current_price,
                mu = gbm_params['mu_daily'],
                sigma = gbm_params['sigma_daily'],
                T = forecast_days,
                dt=1.0,
                n_simulations=n_simulations
            )
        
        percentiles = [5,10,25,40,50,60,75,90,95]
        forecast_stats = {}
        for day in range(forecast_days + 1):
            day_prices = simulations[:, day]
            stats_dict = {
                'mean': np.mean(day_prices),
                'median': np.median(day_prices),
                'std': np.std(day_prices)
            }
            for p in percentiles:
                stats_dict[f'p{p}'] = np.percentile(day_prices, p)
            forecast_stats[day] = stats_dict
        
        final_prices = simulations[:, -1]
        final_stats = {
            'mean_price': np.mean(final_prices),
            'median_price': np.median(final_prices),
            'std_price': np.std(final_prices),
            'min_price': np.min(final_prices),
            'max_price': np.max(final_prices),
            'prob_profit': np.mean(final_prices > current_price),
            'prob_loss_5pct': np.mean(final_prices < current_price * 0.95),
            'prob_loss_10pct': np.mean(final_prices < current_price * 0.90),
            'prob_gain_5pct': np.mean(final_prices > current_price * 1.05),
            'prob_gain_10pct': np.mean(final_prices > current_price * 1.10)
        }
        print(f"\n30-Day Stochastic Forecast Results:")
        print(f"  Expected price: ${final_stats['mean_price']:.2f}")
        print(f"  Median price: ${final_stats['median_price']:.2f}")
        print(f"  Price range: ${final_stats['min_price']:.2f} - ${final_stats['max_price']:.2f}")
        print(f"  Probability of profit: {final_stats['prob_profit']:.1%}")
        print(f"  Probability of >5% gain: {final_stats['prob_gain_5pct']:.1%}")
        print(f"  Probability of >5% loss: {final_stats['prob_loss_5pct']:.1%}")
        
        return {
            'simulations': simulations,
            'gbm_params': gbm_params,
            'best_garch': best_garch,
            'forecast_stats': forecast_stats,
            'final_stats': final_stats,
            'forecast_days': forecast_days,
            'n_simulations': n_simulations,
        }
        
    except Exception as e:
        print(f"Error in stochastic analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_fan_chart(analysis_result, stochastic_result):
    try:
        closing_prices = analysis_result['closing_prices']
        trading_days = analysis_result['trading_days']
        symbol = analysis_result['symbol']
        company_name = analysis_result['company_name']
        current_price = analysis_result['current_price']

        simulations = stochastic_result['simulations']
        forecast_stats = stochastic_result['forecast_stats']
        forecast_days = stochastic_result['forecast_days']

        historical_days = len(closing_prices)
        future_days = np.arange(historical_days, historical_days + forecast_days + 1)

        fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(15,12))

        #Probability Bands Plot 1
        ax1.plot(trading_days, closing_prices, 'b-', linewidth=2, label ='Historical Prices')
        ax1.axvline(x=historical_days-1, color = 'red', linestyle = '--', alpha = 0.7, label='Forecast Start')

        percentile_colors = {
            (5,95): ('red', 0.1),
            (10,90): ('orange', 0.15),
            (25,75): ('yellow', 0.25),
            (40,60): ('lightgreen', 0.4)
        }

        for (lower_p, upper_p), (color, alpha) in percentile_colors.items():
            lower_band = [forecast_stats[day][f'p{lower_p}'] for day in range(forecast_days + 1)]
            upper_band = [forecast_stats[day][f'p{upper_p}'] for day in range(forecast_days + 1)]

            ax1.fill_between(future_days, lower_band, upper_band,
                             color = color, alpha=alpha,
                             label = f'{lower_p}-{upper_p}% confidence band')
        median_forecast = [forecast_stats[day]['median'] for day in range(forecast_days + 1)]
        ax1.plot(future_days, median_forecast, 'g--', linewidth=2, label='Median Forecast')

        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'{company_name} ({symbol}) - Fan Chart with Probability Bands')
        ax1.legend(bbox_to_anchor=(1.05,1), loc = 'upper left')
        ax1.grid(True, alpha = 0.3)

        ax2.plot(trading_days, closing_prices, 'b-', linewidth=3, label = 'Historical Prices')
        ax2.axvline(x = historical_days -1, color='red', linestyle = '--', alpha = 0.7)

        n_paths_to_show = min(100, simulations.shape[0])
        for i in range(0, n_paths_to_show, 5):
            ax2.plot(future_days, simulations[i, :], 'gray', alpha=0.1, linewidth=0.5)

        ax2.plot(future_days, median_forecast, 'g--', linewidth=2, label='Median Forecast')
        ax2.axhline(y=current_price, color='blue', linestyle=':', alpha=0.7, label=f'Current Price: ${current_price:.2f}')
        ax2.set_xlabel('Trading Days')
        ax2.set_ylabel('Price ($)')
        ax2.set_title(f'Monte Carlo Simulation Paths (showing {n_paths_to_show} of {simulations.shape[0]})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        plot_forecast_distribution(stochastic_result, analysis_result)

    except Exception as e:
        print(f'Error plotting fan chart: {e}')
        traceback.print_exc()

def plot_forecast_distribution(stochastic_result, analysis_result):
    """
    Plot the distribution of final forecasted prices
    """
    try:
        simulations = stochastic_result['simulations']
        final_stats = stochastic_result['final_stats']
        current_price = analysis_result['current_price']
        symbol = analysis_result['symbol']

        final_prices = simulations[:, -1]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
        
        ax1.hist(final_prices, bins=50, density = True, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(current_price, color='red', linestyle='--', linewidth=2, label=f'Current: ${current_price:.2f}')
        ax1.axvline(final_stats['mean_price'], color='green', linestyle='-', linewidth=2, label=f'Mean: ${final_stats["mean_price"]:.2f}')
        ax1.axvline(final_stats['median_price'], color='orange', linestyle='-', linewidth=2, label = f'Median: ${final_stats['median_price']:.2f}')

        ax1.set_xlabel('Price ($)')
        ax1.set_ylabel('Probability Density')
        ax1.set_title(f'{symbol} - 30-Day Price Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        returns = (final_prices - current_price) / current_price * 100

        ax2.hist(returns, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor ='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax2.axvline(np.mean(returns), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(returns):.1f}%')
        ax2.axvline(np.median(returns), color='orange', linestyle='-', linewidth=2, label=f'Median: {np.median(returns):.1f}%')

        var_5 = np.percentile(returns,5)
        var_1 = np.percentile(returns, 1)
        ax2.axvline(var_5, color='purple', linestyle=':', linewidth=2, label=f'VaR 5%: {var_5:.1f}%')
        ax2.axvline(var_1, color='darkred', linestyle=':', linewidth=2, label=f'VaR 1%: {var_1:.1f}%')

        ax2.set_xlabel("Return (%)")
        ax2.set_ylabel('Probability Density')
        ax2.set_title(f'{symbol} - 30-Day Return Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f'Eror plotting forecast distribution: {e}')
        traceback.print_exc()
    
def enhanced_stock_analysis(symbol, user_prompt=None, company_name=None):
    result = analyze_stock(symbol, user_prompt, company_name)
    if result is None:
        return None
    stochastic_result = stochastic_analysis(result, n_simulations=5000, forecast_days=30)

    if stochastic_result is None:
        return None
    risk_metrics = calculate_comprehensive_risk_metrics(stochastic_result, result)
    if risk_metrics:
        print_risk_report(risk_metrics)

    plot_stock_analysis(result)
    plot_fan_chart(result,stochastic_result)
    if risk_metrics:
        plot_risk_metrics_dashboard(risk_metrics, result, stochastic_result)

    enhanced_result = result.copy()
    enhanced_result['stochastic'] = stochastic_result
    return enhanced_result



def analyze_stock(symbol, user_prompt=None, company_name=None):
    try:
        if not company_name:
            ticker = Ticker(symbol)
            company_name = ticker.quote_type[symbol].get("longName", symbol)
        print(f"You Selected {user_prompt} / {symbol}")

        end_date = datetime.today()
        start_date = end_date - timedelta(days=90)
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)

        if data is None or data.empty:
            print(f"No data found for {ticker}")
            return None
        print(f"Getting sentiment for {symbol}...")
        sentiment_score, sentiment_label, num_articles = get_stock_sentiment(symbol, company_name)

        closing_prices = data['Close'].dropna().values
        if closing_prices.ndim > 1:
            closing_prices = closing_prices.flatten()
        if len(closing_prices) <10:
            print("Not Enough Data for Closing Prices")
            return None
        

        trading_days = np.arange(len(closing_prices))
        best_model = None
        best_model_type = None
        best_model_params = None
        lowest_mse = float('inf')
        models = {}
        #Polynomial Fitting
        for degree in range (1,5):
            try:
                coefficients = np.polyfit(trading_days, closing_prices, degree).flatten()
                poly = np.poly1d(coefficients)
                preds = poly(trading_days)  
                mse = mean_squared_error(closing_prices, preds)

                x = sym.Symbol('x')
                poly_expr = sum(coefficients[i] * x**(degree-i) for i in range(len(coefficients)))
                first_deriv_expr = sym.diff(poly_expr, x)
                second_deriv_expr = sym.diff(first_deriv_expr, x)

                first_deriv_values = [float(first_deriv_expr.subs(x, day)) for day in trading_days]
                second_deriv_values = [float(second_deriv_expr.subs(x, day)) for day in trading_days]
                
                turning_points = []
                for i in range(1, len(first_deriv_values)):
                    if np.sign(first_deriv_values[i-1]) is not np.sign(first_deriv_values[i]):
                        turning_points.append(i)
                
                models[f'Polynomial_deg_{degree}'] = {
                    'model': poly,
                    'predictions': preds,
                    'mse': mse,
                    'type': 'polynomial',
                    'params': {'degree': degree, 'coefficients':coefficients},
                    'first_derivative': first_deriv_values,
                    'second_derivative': second_deriv_values,
                    'turning_points': turning_points
                }

                if mse < lowest_mse:
                    lowest_mse = mse
                    best_model = poly
                    best_model_type = 'polynomial'
                    best_model_params = {
                        'degree': degree, 
                        'coefficients': coefficients,
                        'first_derivative': first_deriv_values,
                        'second_derivative': second_deriv_values,
                        'turning_points': turning_points
                        }
            except Exception as e:
                print(f"Error finding polynomial fit: {e}")
        #Cubic Spline
        try:
            if len(trading_days) >= 10:
                n_folds = min(5, len(trading_days) // 4)
                kf = KFold(n_splits=n_folds, shuffle=False)

                cv_mses = []
                for train_idx, val_idx in kf.split(trading_days):
                    train_days = trading_days[train_idx]
                    train_prices = closing_prices[train_idx]
                    val_days = trading_days[val_idx]
                    val_prices = closing_prices[val_idx]
                    try:
                        cs_temp = CubicSpline(train_days, train_prices)
                        val_preds = cs_temp(val_days)
                        cv_mse = mean_squared_error(val_prices, val_preds)
                        cv_mses.append(cv_mse)
                    except:
                        cv_mses.append(float('inf'))
                cs_mse = np.mean(cv_mses) if cv_mse else float('inf')
                cs = CubicSpline(trading_days, closing_prices)
                cs_preds = cs(trading_days)

                first_deriv_cs = cs.derivative(nu=1)
                second_deriv_cs = cs.derivative(nu=2)

                first_deriv_values = first_deriv_cs(trading_days)
                second_deriv_values = second_deriv_cs(trading_days)

                turning_points = []
                for i in range(len(first_deriv_values)):
                    if np.sign(first_deriv_values[i-1]) is not np.sign(first_deriv_values[i]):
                        turning_points.append(i)
                models['CubicSpline'] = {
                'model': cs,
                'predictions': cs_preds,
                'mse': cs_mse,
                'type': 'cubic_spline',
                'params': {'spline_object': cs},
                'first_derivative': first_deriv_values,
                'second_derivative': second_deriv_values,
                'turning_points': turning_points
            }

            if cs_mse < lowest_mse:
                lowest_mse = cs_mse
                best_model = cs
                best_model_type = 'cubic_spline'
                best_model_params = {
                    'spline_object': cs, 
                    'first_derivative': first_deriv_values,
                    'second_derivative': second_deriv_values,
                    'turning_points': turning_points
                    }
        except Exception as e:
            print(f"Error fitting Cubic Spline: {e}")
                    
        #B-Spline
        smoothing_factors = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

        for s_factor in smoothing_factors:
            try:
                k=min(3, len(trading_days) - 1)

                if len(trading_days) >= 10:
                    cv_mses = []
                    for train_idx, val_idx in kf.split(trading_days):
                        train_days = trading_days[train_idx]
                        train_prices = closing_prices[train_idx]
                        val_days = trading_days[val_idx]
                        val_prices = closing_prices[val_idx]
                        try:
                            bspline_temp = splrep(train_days, train_prices, s=s_factor, k=k)
                            val_preds = splev(val_days, bspline_temp)
                            cv_mse = mean_squared_error(val_prices, val_preds)
                            cv_mses.append(cv_mse)
                        except:
                            cv_mses.append(float('inf'))

                    bspline_mse = np.mean(cv_mses) if cv_mses else float('inf')
                else:  
                    bspline_data = splrep(trading_days, closing_prices, s=s_factor,k=k)
                    bspline_preds = splev(trading_days, bspline_data)
                    bspline_mse = mean_squared_error(closing_prices, bspline_preds)

                bspline_data = splrep(trading_days, closing_prices, s=s_factor, k=k)
                bspline_preds = splev(trading_days, bspline_data)

                first_deriv_values = []
                second_deriv_values = []

                for i,day in enumerate(trading_days):
                    if i ==0:
                        h= 0.1
                        first_deriv = (splev(day + h, bspline_data) - splev(day, bspline_data)) / h
                    elif i == len(trading_days) - 1:
                        h = 0.1
                        first_deriv = (splev(day, bspline_data) - splev(day - h, bspline_data)) / h
                    else:
                        h = 0.5
                        first_deriv = (splev(day + h, bspline_data) - splev(day - h, bspline_data)) / (2*h)
                    first_deriv_values.append(first_deriv)

                    if i == 0 or i == len(trading_days) - 1:
                        h = 0.1
                        second_deriv = (splev(day + h, bspline_data) - 2*splev(day, bspline_data) + splev(day - h, bspline_data)) / (h**2)
                    else:
                        h = 0.5
                        second_deriv = (splev(day + h, bspline_data) - 2*splev(day, bspline_data) + splev(day - h, bspline_data)) / (h**2)
                    second_deriv_values.append(second_deriv)

                turning_points = []
                for i in range(1, len(first_deriv_values)):
                    if np.sign(first_deriv_values[i-1]) is np.sign(first_deriv_values[i]):
                        turning_points.append(i)
                
                models[f'BSpline_s_{s_factor}'] = {
                    'model': bspline_data,
                    'predictions': bspline_preds,
                    'mse': bspline_mse,
                    'type': 'bspline',
                    'params': {'smoothing': s_factor, 'bspline_data': bspline_data},
                    'first_derivative': first_deriv_values,
                    'second_derivative': second_deriv_values,
                    'turning_points': turning_points
                }

                if bspline_mse < lowest_mse:
                   lowest_mse = bspline_mse
                   best_model = bspline_data
                   best_model_type = 'bspline'
                   best_model_params = {
                       'smoothing': s_factor, 
                       'bspline_data': bspline_data,
                       'first_derivative': first_deriv_values,
                       'second_derivative': second_deriv_values,
                       'turning_points': turning_points
                       }
            except Exception as e:
                print(f"B-Spline Calculation Failed {e}")
                

        if best_model is None:
            print(f"Could not find fit for {symbol}")
            return None
            
        current_price = float(closing_prices[-1])

        if best_model_type == 'polynomial':
            predicted_30_day = float(best_model(len(closing_prices)+30))

        elif best_model_type == 'cubic_spline':
            last_slope = best_model_params['first_derivative'][-1]
            max_daily_change = 0.05
            capped_slope = np.clip(last_slope, -max_daily_change * current_price, max_daily_change * current_price)
            predicted_30_day = current_price + (capped_slope * 30)

        elif best_model_type == 'bspline':
            try:
                predicted_30_day = float(splev(len(closing_prices) + 30, best_model))
                change_pct = abs((predicted_30_day - current_price) / current_price)
                if change_pct > 2.0:
                    raise ValueError("Extreme prediction detected")
            except:
                last_slope = best_model_params['first_derivative'][-1]
                max_daily_change = 0.05
                capped_slope = np.clip(last_slope, -max_daily_change * current_price, max_daily_change * current_price)
                predicted_30_day = current_price + (capped_slope * 30)

        pct_change = ((predicted_30_day - current_price) / current_price) * 100

        return {
            'symbol': symbol,
            'company_name': company_name,
            'best_model_type': best_model_type,
            'best_model': best_model,
            'best_model_params': best_model_params,
            'lowest_mse': lowest_mse,
            'all_models': models,
            'current_price': current_price,
            'predicted_30_day': predicted_30_day,
            'pct_change': pct_change,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'num_articles': num_articles,
            'closing_prices': closing_prices,
            'trading_days': trading_days,
            'slope': best_model_params['first_derivative'][-1],
            'curvature': best_model_params['second_derivative'][-1],
            'turning_points': best_model_params['turning_points'],
            'avg_slope': np.mean(best_model_params['first_derivative']),
            'slope_volatility': np.std(best_model_params['first_derivative']),
            'trend_direction': 'bullish' if best_model_params['first_derivative'][-1] > 0 else 'bearish'
        }
    except Exception as e:
        print(f"Error Finding the Best Model: {e}")
        return None
    






def residual_bootstrap_analysis(analysis_result, n_bootstrap=1000, confidence_levels=[90,95]):
    """
    
    
    """
    try:
        closing_prices = analysis_result['closing_prices']
        trading_days = analysis_result['trading_days']
        best_model = analysis_result['best_model']
        best_model_type = analysis_result['best_model_type']
        best_model_params = analysis_result['best_model_params']

        if best_model_type == 'polynomial':
            fitted_values = best_model(trading_days)
        elif best_model_type == 'cubic_spline':
            fitted_values = best_model(trading_days)
        elif best_model_type == 'bspline':
            fitted_values = best_model(trading_days)

        residuals = closing_prices - fitted_values
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)

        future_days = 30
        future_trading_days = np.arange(len(closing_prices), len(closing_prices) + future_days)
        bootstrap_predictions = []

        for i in range(n_bootstrap):
            bootstrap_residuals = np.random.choice(residuals, size = future_days, replace = True)

            if best_model_type == 'polynomial':
                baseline_pred = best_model(future_trading_days)
            elif best_model_type == 'cubic_spline':
                last_slope = best_model_params['first_derivative'][-1]
                last_price = closing_prices[-1]
                baseline_pred = np.array([last_price + last_slope * (day - trading_days[-1]) for day in future_trading_days])
            elif best_model_type == 'bspline':
                try:
                    baseline_pred = splev(future_trading_days, best_model)
                except:
                    last_slope = best_model_params['first_derivative'][-1]
                    last_price = closing_prices[-1]
                    baseline_pred = np.array([last_price + last_slope * (day - trading_days[-1]) for day in future_trading_days])
            
            bootstrap_path = baseline_pred + bootstrap_residuals
            bootstrap_predictions.append(bootstrap_path)
        bootstrap_predictions = np.array(bootstrap_predictions)

        prediction_intervals = {}

        for conf_level in confidence_levels:
            alpha = (100 - conf_level) / 2
            lower_percentile = alpha
            upper_percentile = 100 - alpha

            lower_bound = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(bootstrap_predictions, upper_percentile, axis = 0)
            median_pred = np.percentile(bootstrap_predictions, 50, axis = 0)

            prediction_intervals[f'{conf_level}%'] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'median_prediction': median_pred,
                'interval_width': upper_bound - lower_bound
            }
        final_day_predictions = bootstrap_predictions[:, -1]

        bootstrap_summary = {
            'mean_30_day': np.mean(final_day_predictions),
            'median_30_day': np.median(final_day_predictions),
            'std_30_day': np.std(final_day_predictions),
            'min_30_day': np.min(final_day_predictions),
            'max_30_day': np.max(final_day_predictions)
        }
        print(f"\nBootstrap Results (30-day Prediction)")
        print(f"Mean: ${bootstrap_summary['mean_30_day']:.2f}")
        print(f"Median: ${bootstrap_summary['median_30_day']:.2f}")
        print(f"Standard Deviation: ${bootstrap_summary['std_30_day']:.2f}")
        print(f"Range: ${bootstrap_summary['min_30_day']:.2f} - ${bootstrap_summary['max_30_day']:.2f}")

        for conf_level in confidence_levels:
            interval = prediction_intervals[f'{conf_level}%']
            final_upper = interval['upper_bound'][-1]
            final_lower = interval['lower_bound'][-1]
            print(f" {conf_level}% \t${final_upper:.2f} - ${final_lower:.2f} (Lower - Upper CI)")
        
        return {
            'residuals': residuals,
            'residual_stats':{
                'mean': residual_mean,
                'std': residual_std,
                'min': np.min(residuals),
                'max': np.max(residuals)
            },
            'bootstrap_predictions': bootstrap_predictions,
            'prediction_intervals': prediction_intervals,
            'bootstrap_summary': bootstrap_summary,
            'future_trading_days': future_trading_days,
            'fitted_values': fitted_values
        }
    except Exception as e:
        print(f"Error Analyzing Residuals with Bootstrapping: {e}")
        return None







def plot_stock_analysis(result):
    try:
        symbol=result['symbol']
        company_name = result['company_name']
        closing_prices = result['closing_prices']
        trading_days = result['trading_days']
        best_model = result['best_model']
        best_model_type = result['best_model_type']
        deriv_values = result['best_model_params']['first_derivative']
        critical_points = result['best_model_params']['turning_points']
        lowest_mse = result['lowest_mse']


        plt.figure(figsize = (16,12))
        plt.subplot(2,2,1)
        plt.plot(trading_days, closing_prices, label = "Actual Prices", linewidth=2)
        if best_model_type == 'polynomial':
            fitted_values = best_model(trading_days)
        elif best_model_type == 'cubic_spline':
            fitted_values = best_model(trading_days)
        elif best_model_type == 'bspline':
            fitted_values = splev(trading_days, best_model)

        #Best Model
        plt.plot(trading_days, fitted_values,'g-' ,label =f"Best Fit:{best_model_type}", linewidth=2)
        plt.xlabel("Trading Days")
        plt.ylabel("Price ($)")
        plt.title(f"{company_name} ({symbol}) - Stock Price Analysis")
        plt.legend()
        plt.grid(True, alpha=0.3)

        #Derivative
        plt.subplot(2,2,2)
        plt.plot(trading_days, deriv_values, 'r--', label = "Derivative", linewidth=2)
        plt.axhline(0, color='red', linestyle=':', alpha =0.7)
        plt.xlabel("Trading Days")
        plt.ylabel("Rate of Change")
        plt.title("Derivative of Function")

        for cr in critical_points:
            if cr < len(trading_days):
                plt.axvline(cr, color ='red', alpha=0.6, linestyle = '--')
                plt.scatter(cr, deriv_values[cr], color = 'red', s=50, zorder = 5)

        plt.legend()
        plt.grid(True, alpha = 0.3)


        plt.subplot(2,2,3)
        model_names = []
        model_mses = []
        colors = ['blue','purple','orange', 'red', 'green']

        for i, (model_name, model_info) in enumerate(result['all_models'].items()):
            model_names.append(model_name[:15])
            model_mses.append(model_info['mse'])
            
        if model_names and model_mses:
            bars = plt.bar(range(len(model_names)), model_mses, color=colors[:len(model_names)], alpha = 0.7)

            for i, (bar , mse) in enumerate(zip(bars, model_mses)):
                    plt.text(i, mse + max(model_mses) * 0.01, f'{mse:.2e}', 
                             ha = 'center', va= 'bottom', fontsize = 8, rotation=45)
                    
            plt.xticks(range(len(model_names)), model_names, rotation = 45, ha='right')
            plt.ylabel("MSE")
            plt.title("Model Comparison (Lowest MSE = Better)")
            plt.yscale('log')
            plt.grid(True, alpha=0.3)


        plt.subplot(2,2,4)
        plt.axis("off")
        info_text = f"""
        Stock: {symbol}
        Company: {company_name}

        Best Model: {best_model_type}
        MSE: {lowest_mse:.6f}

        Current Price: ${result['current_price']:.2f}
        30-Day Prediction: ${result['predicted_30_day']:.2f}
        Expected Change: {result['pct_change']:+.2f}%

        Sentiment: {result['sentiment_label']}
        Sentiment Score: {result['sentiment_score']:.3f}
        News Articles: {result['num_articles']}

        Critical Points: {len(critical_points)}
        """

        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.subplots_adjust(left = 0.08, bottom = 0.08, right = 0.95, top = 0.95,
                            wspace = 0.25, hspace = 0.35)
        plt.show(block = False)
        plt.pause(0.1)
        
    except Exception as e:
        print(f"Error plotting stock analysis: {e}")
        traceback.print_exc()

def calculate_comprehensive_risk_metrics(stochastic_result, analysis_result,risk_free_rate = 0.0418):
    try:
        simulations = stochastic_result['simulations']
        current_price = analysis_result['current_price']
        forecast_days = stochastic_result['forecast_days']
        symbol = analysis_result['symbol']

        final_prices = simulations[:, -1]
        returns = (final_prices - current_price) / current_price
        returns_pct = returns * 100
        
        daily_rf_rate = (1 + risk_free_rate) ** (1/252) -1
        var_levels = [1,5,10]
        var_metrics = {}
        for level in var_levels:
            var_value = np.percentile(returns_pct, level)
            var_price = current_price * (1 + var_value/100)
            var_metrics[f'VaR_{level}%'] = {
                'return_pct': var_value,
                'price': var_price,
                'loss_amount': var_price - current_price
            }

        cvar_metrics = {}
        for level in var_levels:
            threshold = np.percentile(returns_pct, level)
            tail_losses = returns_pct[returns_pct <= threshold]
            cvar_value = np.mean(tail_losses) if len(tail_losses) > 0 else threshold
            cvar_price = current_price * (1 + cvar_value/100)
            cvar_metrics[f'CVaR_{level}%'] = {
                'return_pct': cvar_value,
                'price': cvar_price,
                'loss_amount': cvar_price - current_price,
            }
        prob_metrics = {
            'prob_profit': np.mean(final_prices > current_price),
            'prob_loss_5pct': np.mean(final_prices < current_price * 0.95),
            'prob_loss_10pct': np.mean(final_prices < current_price * 0.90),
            'prob_loss_20pct': np.mean(final_prices < current_price * 0.80),
            'prob_gain_5pct': np.mean(final_prices > current_price * 1.05),
            'prob_gain_10pct': np.mean(final_prices > current_price * 1.10),
            'prob_gain_20pct': np.mean(final_prices > current_price * 1.20),
            'prob_gain_50pct': np.mean(final_prices > current_price * 1.50),
        }

        max_drawdowns = []
        for sim_path in simulations:
            running_max = np.maximum.accumulate(sim_path)
            drawdown = (sim_path - running_max) / running_max
            max_dd = np.min(drawdown)
            max_drawdowns.append(max_dd)

        max_drawdown_metrics = {
            'avg_max_drawdown': np.mean(max_drawdowns) * 100,
            'worst_max_drawdown': np.min(max_drawdowns) * 100,
            'median_max_drawdown': np.median(max_drawdowns) * 100,
            '95th_percentile_drawdown': np.percentile(max_drawdowns, 5) * 100
        }

        daily_returns_all = []
        for sim_path in simulations:
            daily_rets = np.diff(sim_path) / sim_path[:-1]
            daily_returns_all.append(daily_rets)

        daily_returns_all = np.array(daily_returns_all)
        avg_daily_return = np.mean(daily_returns_all)
        std_daily_return = np.std(daily_returns_all)

        sharpe_ratio = (avg_daily_return - daily_rf_rate) / std_daily_return if std_daily_return > 0 else 0
        annualized_sharpe = sharpe_ratio * np.sqrt(252)

        downside_returns = daily_returns_all[daily_returns_all < daily_rf_rate]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_daily_return

        sortino_ratio = (avg_daily_return - daily_rf_rate) / downside_std if downside_std > 0 else 0
        annualized_sortino = sortino_ratio * np.sqrt(252)

        expected_return = np.mean(returns_pct)
        expected_loss_if_negative = np.mean(returns_pct[returns_pct < 0]) if np.any(returns_pct < 0) else 0
        expected_gain_if_positive = np.mean(returns_pct[returns_pct < 0]) if np.any(returns_pct > 0) else 0

        gain_to_loss_ratio = abs(expected_gain_if_positive / expected_loss_if_negative) if expected_loss_if_negative is not 0 else float('inf')

        volatility_metrics = {
            'return_std': np.std(returns_pct),
            'return_variance': np.var(returns_pct),
            'annualized_volatility': np.std(returns_pct) * np.sqrt(252/forecast_days),
            'coefficient_of_variation': np.std(returns_pct) / np.mean(returns_pct) if np.mean(returns_pct) is not 0 else float('inf')
        }
        risk_metrics = {
            'symbol': symbol,
            'current_price': current_price,
            'forecast_days': forecast_days,
            'n_simulations': len(simulations),

            'var': var_metrics,
            'cvar': cvar_metrics,
            'probability_targets': prob_metrics,
            'max_drawdown': max_drawdown_metrics,

            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': annualized_sortino,
            'daily_sharpe': sharpe_ratio,
            'daily_sortino': sortino_ratio,

            'expected_return_pct': expected_return,
            'expected_loss_if_negative': expected_loss_if_negative,
            'expected_gain_if_positive': expected_gain_if_positive,
            'gain_to_loss_ratio': gain_to_loss_ratio,

            'volatility': volatility_metrics,

            'return_distribution': {
                'mean': np.mean(returns_pct),
                'median': np.median(returns_pct),
                'std': np.std(returns_pct),
                'skewness': pd.Series(returns_pct).skew(),
                'kurtosis': pd.Series(returns_pct).kurtosis()
            }
        }
        return risk_metrics
    except Exception as e:
        print(f'Error calculating risk metrics: {e}')
        traceback.print_exc()
        return None

def print_risk_report(risk_metrics):
    if not risk_metrics:
        print("No Risk Metrics Available")
        return
    try:
        symbol = risk_metrics.get('symbol','Unknown')
        current_price = risk_metrics.get('current_price',0)
        forecast_days = risk_metrics.get('forecast_days',0)

        print(f"\n{'='*80}")
        print(f"Comprehensive Risk Analysis Report: {symbol}")
        print(f"\n{'='*80}")
        print(f"Current Price: ${current_price:.2f} | Forecast Horizon: {forecast_days} days")
        print(f"Simulations: {risk_metrics.get('n_simulations',0):,}\n")
        if 'var' in risk_metrics:
            print(f"{'Value at Risk (VaR)':-^80}")
            print(f"{'Confidence':<15} {'Return %':<15} {'Price':<15} {'Loss Amount'}")
            print("-" * 80)
            for level, metrics in risk_metrics['var'].items():
                print(f"{level:<15} {metrics['return_pct']:>12.2f}% ${metrics['price']:>12.2f} ${metrics['loss_amount']:>12.2f}")
        if 'cvar' in risk_metrics:
            print(f"\n{'Conditional Value at Risk (CVaR / Expected Shortfall)':-^80}")
            print(f"{'Confidence':<15} {'Return %':<15} {'Price':<15} {'Loss Amount'}")
            print("-" * 80)
            for level, metrics in risk_metrics['cvar'].items():
                print(f"{level:<15} {metrics['return_pct']:>12.2f}% ${metrics['price']:>12.2f} ${metrics['loss_amount']:>12.2f}")
        
        if 'probability_targets' in risk_metrics:
            print(f"\n{'Probability of Targets':-^80}")
            prob = risk_metrics['probability_targets']
            print(f"Probability of ANY profit:        {prob['prob_profit']:>6.1%}")
            print(f"Probability of >5% gain:          {prob['prob_gain_5pct']:>6.1%}")
            print(f"Probability of >10% gain:         {prob['prob_gain_10pct']:>6.1%}")
            print(f"Probability of >20% gain:         {prob['prob_gain_20pct']:>6.1%}")
            print(f"Probability of >5% loss:          {prob['prob_loss_5pct']:>6.1%}")
            print(f"Probability of >10% loss:         {prob['prob_loss_10pct']:>6.1%}")
            print(f"Probability of >20% loss:         {prob['prob_loss_20pct']:>6.1%}")
        if 'max_drawdown' in risk_metrics:
            print(f"\n{'Maximum Drawdown':-^80}")
            dd = risk_metrics['max_drawdown']
            print(f"Average Max Drawdown:             {dd.get('avg_max_drawdown',0):>6.2f}%")
            print(f"Median Max Drawdown:              {dd.get('median_max_drawdown',0):>6.2f}%")
            print(f"Worst Max Drawdown:               {dd.get('worst_max_drawdown',0):>6.2f}%")
            print(f"95th Percentile Drawdown:         {dd.get('95th_percentile_drawdown',0):>6.2f}%")
        
        print(f"\n{'Risk-Adjusted Performance':-^80}")
        print(f"Sharpe Ratio (Annualized):        {risk_metrics.get('sharpe_ratio',0):>6.3f}")
        print(f"Sortino Ratio (Annualized):       {risk_metrics.get('sortino_ratio',0):>6.3f}")
        print(f"Expected Return:                  {risk_metrics.get('expected_return_pct',0):>6.2f}%")

        gain_loss = risk_metrics.get('gain_to_loss_ratio',0)
        if gain_loss == float('inf'):
            print(f"Gain/Loss Ratio:                  {'∞':>6}")
        else:
            print(f"Gain/Loss Ratio:                  {gain_loss:>6.2f}x")
            
        if 'volatility' in risk_metrics:
            print(f"\n{'Volatility Metrics':-^80}")
            vol = risk_metrics['volatility']
            print(f"Return Std Dev:                   {vol.get('return_std', 0):>6.2f}%")
            print(f"Annualized Volatility:            {vol.get('annualized_volatility', 0):>6.2f}%")
        
        print(f"\n{'='*80}\n")
    except Exception as e:
        print(f"Error Printing Risk Metrics: {e}")
        traceback.print_exc()


def plot_risk_metrics_dashboard(risk_metrics, analysis_result, stochastic_result):
    try:
        simulations = stochastic_result['simulations']
        current_price = analysis_result['current_price']
        symbol = analysis_result['symbol']

        final_prices = simulations[:, -1]
        returns_pct = ((final_prices - current_price) / current_price) * 100

        fig = plt.figure(figsize=(18,12))
        gs = fig.add_gridspec(3, 3, hspace = 0.3, wspace = 0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        var_levels = ['VaR_1%', 'VaR_5%', 'VaR_10%']
        var_values = [risk_metrics['var'][k]['return_pct'] for k in var_levels]
        cvar_values = [risk_metrics['cvar'][k.replace('VaR', 'CVaR')]['return_pct'] for k in var_levels]

        x = np.arange(len(var_levels))
        width = 0.35
        ax1.bar(x - width/2, var_values, width, label = 'VaR', alpha=0.8, color = 'orange')
        ax1.bar(x + width/2, cvar_values, width, label = 'CVaR', alpha = 0.8, color = 'red')
        ax1.set_xlabel('Confidence Level')
        ax1.set_ylabel('Return (%')
        ax1.set_title('VaR vs CVaR Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['99%', '95%', '90%'])
        ax1.legend()
        ax1.grid(True, alpha = 0.3)
        ax1.axhline(y=0, color = 'black', linestyle='--', linewidth = 0.8)

        ax2 = fig.add_subplot(gs[0,1])
        prob = risk_metrics['probability_targets']
        targets = ['5% Gain', '10% Gain', '20% Gain', '5% Loss', '10% Loss', '20% Loss']
        probs = [prob['prob_gain_5pct'], prob['prob_gain_10pct'], prob['prob_gain_20pct'], 
                 prob['prob_loss_5pct'], prob['prob_loss_10pct'], prob['prob_loss_20pct']]
        colors_probs = ['green', 'darkgreen', 'lime', 'orange', 'red', 'darkred']

        bars = ax2.barh(targets, probs, color=colors_probs, alpha=0.7)
        ax2.set_xlabel('Probability')
        ax2.set_title('Target Probability Distribution')
        ax2.set_xlim(0, 1)
        for i, (bar,p) in enumerate(zip(bars, probs)):
            ax2.text(p + 0.02, i,f'{p:.1%}', va='center')
        ax2.grid(True, alpha=0.3, axis='x')

        ax3 = fig.add_subplot(gs[0,2])
        max_drawdowns = []
        for sim_path in simulations:
            running_max = np.maximum.accumulate(sim_path)
            drawdown = (sim_path - running_max) / running_max
            max_dd = np.min(drawdown) * 100
            max_drawdowns.append(max_dd)

        ax3.hist(max_drawdowns, bins=50, alpha = 0.7, color='purple', edgecolor='black')
        ax3.axvline(risk_metrics['max_drawdown']['avg_max_drawdown'], color='red', linestyle='--',linewidth=2, label='Average')
        ax3.axvline(risk_metrics['max_drawdown']['worst_max_drawdown'], color='darkred', linestyle=':',linewidth=2,label = 'Worst')
        ax3.set_xlabel('Max Drawdown (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Maximum Drawdown Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, :])
        ax4.hist(returns_pct, bins=100, alpha=0.6, color='skyblue', edgecolor='black', density=True)
        ax4.axvline(0, color='black', linestyle='--', linewidth=2, label='Break-even')
        ax4.axvline(risk_metrics['expected_return_pct'], color ='green',
                     linestyle='-', linewidth=2, label=f'Expected: {risk_metrics["expected_return_pct"]:.2f}')
        for level in ['VaR_1%', 'VaR_5%', 'VaR_10%']:
            var_val = risk_metrics['var'][level]['return_pct']
            ax4.axvline(var_val, color='red',linestyle=':', alpha=0.7, linewidth=1.5)
            ax4.text(var_val, ax4.get_ylim()[1]*0.9, level, rotation=90, va='top', fontsize=8)

        ax4.set_xlabel('Return (%)')
        ax4.set_ylabel('Probability Density')
        ax4.set_title(f'{symbol} - Return Distribution with Risk Metrics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[2,0])
        metrics_names = ['Sharpe\nRatio', 'Sortino\nRatio', 'Gain/Loss\nRatio']
        metrics_values = [risk_metrics['sharpe_ratio'],
                          risk_metrics['sortino_ratio'],
                          min(risk_metrics['gain_to_loss_ratio'],5)]
        color_metrics= ['blue' if v> 0 else 'red' for v in metrics_values]
        bars = ax5.bar(metrics_names, metrics_values, color=color_metrics, alpha=0.7)
        ax5.set_ylabel('Ratio Value')
        ax5.set_title('Risk-Adjusted Performance')
        ax5.axhline(y=0, color='black',linestyle='-',linewidth=0.8)
        ax5.grid(True, alpha=0.3, axis='y')
        for bar,val in zip(bars, metrics_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                     f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top')
            
        ax6 = fig.add_subplot(gs[2,1:])
        avg_path = np.mean(simulations,axis=0)
        running_max = np.maximum.accumulate(avg_path)
        drawdown_path = (avg_path - running_max) / running_max * 100
        
        days = np.arange(len(avg_path))
        ax6.fill_between(days, drawdown_path, 0,alpha=0.5,color='red', label='Drawdown')
        ax6.plot(days, drawdown_path, color='darkred', linewidth=2)
        ax6.set_xlabel('Trading Days')
        ax6.set_ylabel('Drawdown (%)')
        ax6.set_title('Average Drawdown Path')
        ax6.legend()
        ax6.grid(True, alpha = 0.3)
        ax6.axhline(y=0, color='black', linestyle = '-', linewidth=0.8)

        plt.suptitle(f'{symbol} - Comprehensive Risk Metrics Dashboard', fontsize=16, fontweight='bold', y=0.995)
        plt.show()
    except Exception as e:
        print(f"Failed to plot Risk Metrics Dashboard: {e}")
        traceback.print_exc()

            





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

            if isinstance(result, dict):
                    print(f"Symbol: {result.get('symbol', 'Missing')}")
                    print(f"Best model type: {result.get('best_model_type', 'Missing')}")
                    print(f"MSE: {result.get('lowest_mse', 'Missing')}")
            try:
                plot_stock_analysis(result)
            except Exception as e:
                print(f"Plotting failed with error: {e}")
                traceback.print_exc()

            do_stochastic = input("Run stochastic analysis? (y/n, default=y): ").strip().lower()
            if do_stochastic is not 'n':
                try:
                    stochastic_result= stochastic_analysis(result,n_simulations=3000, compare_models=True)
                    if stochastic_result:
                        if 'model_comparison' in stochastic_result:
                            best_model_name = stochastic_result['model_comparison'].iloc[0]['name']
                            best_aic = stochastic_result['model_comparison'].iloc[0]['aic']
                            print(f"Best GARCH Model: {best_model_name} (AIC: {best_aic}")
                        risk_metrics = calculate_comprehensive_risk_metrics(stochastic_result, result)
                        if risk_metrics:
                            print_risk_report(risk_metrics)
                            plot_risk_metrics_dashboard(risk_metrics, result, stochastic_result)
                        plot_fan_chart(result, stochastic_result)
                    else:
                        print("Stochastic Analysis Failed")
                except Exception as e:
                    print(f"Stochastic analysis error: {e}")
                    traceback.print_exc()
            try:
                bootstrap_result = residual_bootstrap_analysis(result)
                if bootstrap_result:
                    print("Bootstrap Analysis Complete")
            except Exception as e:
                print("Error using Bootstrap Analysis: {e}")

        else:
            print("Analysis failed - no result returned")
        

def auto_stock_analysis():
    global global_analysis_results
    analysis_results=[]
    test_stocks = all_stocks[:20] #Fix Later

    for i, symbol in enumerate(test_stocks, 1):
        print(f"Processing {symbol}")
        try:
            result = analyze_stock(symbol)
            if result:
                bootstrap_result = residual_bootstrap_analysis(result, n_bootstrap=500)
                if bootstrap_result:
                    result['bootstrap'] = bootstrap_result
                analysis_results.append(result)
                print(f"Successfully analyzed {symbol}")
            time.sleep(0.5)
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            continue
    
    if not analysis_results:
        print("No successful analysis completed")
        return
    analysis_results.sort(key=lambda x: x['lowest_mse'])
    global_analysis_results = analysis_results
    print(f"\n{'='*80}")
    print("TOP 20 STOCKS RANKED BY MODEL FIT (Lower MSE = Better)")
    print(f"{'='*80}")
    print(f"{'Rank':<4} {'Symbol':<8} {'Company':<30} {'Model':<15} {'MSE':<12} {'Sentiment':<10} {'Price Change'}")
    print(f"{'-'*80}")
    
    for rank, result in enumerate(analysis_results[:20],1):
        current = result['current_price']
        predicted = result['predicted_30_day']
        pct_change = result['pct_change']

        company_short = result['company_name'][:28] + ".." if len(result['company_name']) > 30 else result['company_name']

        print(f"{rank:<4} {result['symbol']} {company_short:<30} {result['best_model_type']:<15} "
              f"MSE: {result['lowest_mse']} Sentiment: {result['sentiment_score']} Sentiment Label: {result['sentiment_label']}"
              f"${result['current_price']:2f} → ${result['predicted_30_day']:2f} ({pct_change:+6.1f}%)")
        
    print(f"{'-'*80}")
    print("Enter a number (1-20) to plot that stock's analysis, or 'q' to return to main menu")


        

def plot_selected_stock():
    """Allows User to select and plot a Stock"""
    global global_analysis_results

    if not global_analysis_results:
        print("No Analysis Results Available: Run Auto Analysis First")
        return
    while True:
        try:
            choice = input("\nEnter Stock Number (1,20) to plot or 'q' to quit").strip().lower()
            if choice == 'q':
                break
            stock_num = int(choice)
            if 1<= stock_num <= len(global_analysis_results):
                result = global_analysis_results[stock_num-1]
                print(f"Plotting Graphs for {result['company_name']} ({result['symbol']})")
                if 'bootstrap' not in result:
                    print("Running Bootstrap Analysis")
                    bootstrap_result = residual_bootstrap_analysis(result)
                    if bootstrap_result:
                        result['bootstrap'] = bootstrap_result
                plot_stock_analysis(result)
            else:
                print(f"Please enter a number 1 and {min(20,len(global_analysis_results))}")
        
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except Exception as e:
            print(f"Error: {e}")

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
    global global_analysis_results

    while True:
        print(f"\n{'='*50}")
        print("Stock Risk Analysis + Predictor")
        print(f"{'='*50}")
        print("1 For Manually Checking Stocks")
        print("2 For Automatic Top Stocks")
        print("3 For Plotting Graphs")
        print("Q to Quit")
        choice = input("Choose Option 1 or 2 ").strip()
        if choice == "1":
            manual_analysis()   
        elif choice == "2":
            auto_stock_analysis()
        elif choice =="3":
            plot_selected_stock()
        elif choice.lower() =="q":
            break
        else:
            print("Invalid choice")
if __name__ == "__main__":
    main() 
