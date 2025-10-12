import numpy as np
import pandas as pd
from stochastic_models import stochastic_analysis

def create_model_predictor(symbol, forecast_horizon =252):
    def predict_risk_metrics(historical_data, train_end_idx):
        try:
            train_data = historical_data.iloc[:train_end_idx+1].copy()
            
            print(f"  Running model on {len(train_data)} days...")
    
            closing_prices = train_data['Close'].values
            trading_days = np.arange(len(closing_prices))

            returns = np.diff(closing_prices) / closing_prices[:-1]
            lookback_days = min(504, len(returns))
            recent_returns = returns[-lookback_days:]
            recent_prices = closing_prices[-lookback_days-1:]

            historical_vol = np.std(recent_returns) * np.sqrt(252) * 100
            historical_mean_return = np.mean(recent_returns) * 252

            analysis_result = {
                'closing_prices': recent_prices,
                'trading_days': np.arange(len(recent_prices)),
                'symbol': symbol,
                'company_name': symbol,
                'current_price': closing_prices[-1],
                'predicted_30_day': closing_prices[-1] * (1 + historical_mean_return/12),
                'pct_change': (historical_mean_return/12) * 100,
                'best_model': None,
                'best_model_type': 'polynomial',
                'best_model_params': {
                    'first_derivative': np.zeros(len(recent_prices)),
                    'turning_points': []
                },
                'lowest_mse': 0.01,
                'all_models': {},
                'sentiment_label': 'Neutral',
                'sentiment_score': 0.0,
                'num_articles': 0
            }

            stochastic_result = stochastic_analysis(
                analysis_result,
                n_simulations=20000,
                forecast_days=forecast_horizon,
                compare_models=False,
                verbose = False
            )
            
            if stochastic_result is None:
                print("Stochastic analysis failed")
                return None
            
            from plots import calculate_comprehensive_risk_metrics
            risk_metrics = calculate_comprehensive_risk_metrics(
                stochastic_result,
                analysis_result
            )
            
            if risk_metrics is None:
                print("Risk calculation failed")
                return None
            
            daily_rf = (1+0.0418) ** (1/252) - 1
            excess_returns = recent_returns-daily_rf
            historical_sharpe = (np.mean(excess_returns) / np.std(recent_returns)) * np.sqrt(252) if np.std(recent_returns) > 0 else 0

            downside_returns = recent_returns[recent_returns < daily_rf]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(recent_returns)
            historical_sortino = (np.mean(excess_returns) / downside_std) * np.sqrt(252) if downside_std > 0 else 0

            cumulative = np.cumprod(1 + recent_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            historical_max_dd = abs(np.min(drawdown)) * 100

            n_bootstrap = 1000
            horizon_returns = []
            for b in range(n_bootstrap):
                sampled_returns = np.random.choice(recent_returns, size=forecast_horizon, replace=True)
                cumulative_return = (np.prod(1 + sampled_returns) -1) * 100
                horizon_returns.append(cumulative_return)

            var_5_forecast = np.percentile(horizon_returns, 5)
            var_1_forecast = np.percentile(horizon_returns, 1)

            model_vol = risk_metrics['volatility']['annualized_volatility']
            ewma_vol = calculate_ewma_volatility(recent_returns)
            blended_vol = 0.7 * ewma_vol + 0.2 * historical_vol + 0.1 * model_vol

            recent_momentum = np.mean(recent_returns[-63:])
            predicted_sharpe = np.clip(recent_momentum * np.sqrt(252) * 2, -0.5, 1.5)
            predicted_sortino = predicted_sharpe * 1.2
            
            adjusted_sharpe = predicted_sharpe
            adjusted_sortino = predicted_sortino
            bootstrap_drawdowns = []

            for b in range(1000):
                sampled_returns = np.random.choice(recent_returns, size=forecast_horizon, replace=True)
                cumulative = np.cumprod(1 + sampled_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown_path = (cumulative - running_max) / running_max
                max_dd = abs(np.min(drawdown_path)) * 100
                bootstrap_drawdowns.append(max_dd)
            
            bootstrap_dd = np.percentile(bootstrap_drawdowns, 25)
            historical_dd_forecast = historical_max_dd * 0.90
            adjusted_max_dd = min(bootstrap_dd, historical_dd_forecast)

            predictions = {
                'sharpe_ratio': adjusted_sharpe,
                'sortino_ratio': adjusted_sortino,
                'max_drawdown': adjusted_max_dd,
                'volatility': blended_vol,
                'var_5': var_5_forecast,
                'var_1': var_1_forecast,
                'predicted_price': analysis_result['predicted_30_day'],
                'predicted_return_pct': analysis_result['pct_change'],
                'prob_profit': risk_metrics['probability_targets']['prob_profit'],
            }
            
            print(f"Sharpe: {predictions['sharpe_ratio']:.3f}, Vol: {predictions['volatility']:.1f}%")
            
            return predictions
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return predict_risk_metrics
def calculate_ewma_volatility(returns, lambda_param=0.94):
    squared_returns = returns ** 2
    weights = np.array([(1 - lambda_param) * (lambda_param ** i) 
                        for i in range(len(returns))])
    weights = weights[::-1]
    weights = weights / weights.sum()
    
    ewma_var = np.sum(weights * squared_returns)
    ewma_vol = np.sqrt(ewma_var) * np.sqrt(252) * 100
    
    return ewma_vol