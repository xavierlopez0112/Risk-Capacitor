import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import StockBacktester
from model_wrapper import create_model_predictor

def download_data(symbol, period='max'):
    print("Downloading Stock Data")
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    data = pd.DataFrame({
        'Date': hist.index,
        'Close': hist['Close'].values
    }).reset_index(drop=True)

    return data

if __name__ == "__main__":

    symbol = "AAPL"
    
    data = download_data(symbol, period="max")

    TEST_PERIOD = 252
    FORECAST_HORIZON = 252
    
    print("Initializing backtester...")
    backtester = StockBacktester(
        test_period_days=TEST_PERIOD,
        forecast_horizon=FORECAST_HORIZON,
        min_training_days=756
    )
    model_function = create_model_predictor(symbol,forecast_horizon=FORECAST_HORIZON)
    print("Running backtest")
    results = backtester.run_backtest(
        symbol=symbol,
        historical_data=data,
        model_function=model_function,
        overlap_pct=0.5
    )
    
    print("Calculating metrics")
    metrics = backtester.calculate_validation_metrics()
    
    backtester.print_validation_report(metrics)
    
    print("Generating plots")
    backtester.plot_backtest_results()
    
    output_file = f"{symbol}_backtest_results.csv"
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")