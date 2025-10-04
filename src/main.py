from dotenv import load_dotenv
import numpy as np
import os
import traceback
import warnings
import time
from yahooquery import search, Ticker, Screener

load_dotenv()

from src.analyzing_functions import analyze_stock, residual_bootstrap_analysis
from src.plots import plot_fan_chart, plot_stock_analysis, calculate_comprehensive_risk_metrics, print_risk_report, plot_risk_metrics_dashboard
from src.stochastic_models import stochastic_analysis
from src.sentiment_finder import get_stock_sentiment
warnings.filterwarnings('ignore')


s = Screener()

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
            if do_stochastic != 'n':
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
        
def calculate_overall_quality(result):
    mse = result['lowest_mse']
    predicted = result['predicted_30_day']
    current = result['current_price']
    pct_change = abs(result['pct_change'])

    slope_volatility = result.get('slope_volatility', 0)
    sentiment_score = abs(result.get('sentiment_score', 0))
    num_turning_points = len(result.get('turning_points', []))

    if predicted < 0:
        return float('inf')
    if predicted > 500:
        return float('inf')
    if mse < 0.01 and pct_change > 200:
        return float('inf')
    
    mse_component=mse
    if pct_change > 100:
        extreme_penalty = np.exp((pct_change - 100) / 100)
    else:
        extreme_penalty = 1.0

    volatility_penalty = 1 + (slope_volatility / 10)
    sentiment_bonus = 1 - (sentiment_score * 0.1)
    
    if num_turning_points > 5:
        turning_point_penalty = 1 + (num_turning_points - 5) * 0.2
    else:
        turning_point_penalty = 1.0

    quality_score = (mse_component * extreme_penalty * volatility_penalty * turning_point_penalty) / sentiment_bonus
    return quality_score











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
                try:
                    stochastic_result = stochastic_analysis(result, n_simulations=1000,compare_models=False)
                    if stochastic_result:
                        result['stochastic'] = stochastic_result

                        risk_metrics = calculate_comprehensive_risk_metrics(stochastic_result, result)
                        if risk_metrics:
                            result['risk_metrics'] = risk_metrics
                except Exception as e:
                    print(f"Error Using Stochastic Analysis for {symbol}: {e}")
                analysis_results.append(result)
                print(f"Successfully analyzed {symbol}")
            time.sleep(0.5)
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            continue
    seen_symbols = set()
    unique_results = []
    for result in analysis_results:
        if result['symbol'] not in seen_symbols:
            unique_results.append(result)
            seen_symbols.add(result['symbol'])

    print("Calculating Quality Scores")
    for result in unique_results:
        result['quality_score'] = calculate_overall_quality(result)

    unique_results.sort(key=lambda x: x['quality_score'])
    valid_results = [r for r in unique_results if r['quality_score'] != float('inf')]

    global_analysis_results = valid_results

    print(f"\n{'='*80}")
    print("TOP 20 STOCKS RANKED BY PREDICTION QUALITY ")
    print(f"{'='*80}")
    print(f"{'#':<4} {'Symbol':<8} {'Company':<30} {'Current':<11} {'Predicted':<13} {'Change':<10} {'MSE':<10} {'Risk':<10}")
    print(f"{'-'*80}")
    
    for rank, result in enumerate(valid_results[:20],1):
        current = result['current_price']
        predicted = result['predicted_30_day']
        pct_change = result['pct_change']

        company_short = result['company_name'][:28] + ".." if len(result['company_name']) > 30 else result['company_name']

        if 'risk_metrics' in result:
            sharpe = result['risk_metrics'].get('sharpe_ratio', 0)
            risk_indicator = f"SR:{sharpe:>5.2f}"
        else:
            risk_indicator = "N/A"
        change_indicator = "UP" if pct_change > 0 else "DOWN"

        print(f"{rank:<4} {result['symbol']} {company_short:<30} "
              f"${current:>8.2f} --> ${predicted:>8.2f}"
              f"{change_indicator} {pct_change:>7.1f}%  "
              f"{result['lowest_mse']:>8.4f}    {risk_indicator:<10}")
        
    print(f"{'-'*80}")
    input("Press Enter to Continue...")



        

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
