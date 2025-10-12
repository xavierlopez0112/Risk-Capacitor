import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats

class StockBacktester:
    def __init__(self, test_period_days=126, forecast_horizon = None, min_training_days=756):
        self.test_period_days = test_period_days
        self.forecast_horizon = forecast_horizon if forecast_horizon else test_period_days
        self.min_training_days = min_training_days
        self.results = []
        self.risk_metric_mode = (forecast_horizon is None or forecast_horizon >= 63)

    def calculate_realized_risk_metrics(self, prices, risk_free_rate = 0.0418):
        if len(prices) < 2:
            return None
        returns = np.diff(prices) / prices[:-1]
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = returns - daily_rf
        sharpe = np.mean(excess_returns) / np.std(returns) if np.std(returns) > 0 else 0
        annualized_sharpe = sharpe * np.sqrt(252)

        downside_returns = returns[returns < daily_rf]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        sortino = np.mean(excess_returns) / downside_std if downside_std > 0 else 0
        annualized_sortino = sortino * np.sqrt(252)

        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100

        annualized_vol = np.std(returns) * np.sqrt(252) * 100
        total_return = (prices[-1] - prices[0]) / prices[0] * 100
        return {
            'sharpe_ratio': annualized_sharpe,
            'sortino_ratio': annualized_sortino,
            'max_drawdown_pct': max_drawdown,
            'annualized_volatility': annualized_vol,
            'total_return_pct': total_return,
            'avg_daily_return': np.mean(returns) * 100,
            'n_observations': len(returns)
        }
    def generate_test_windows(self, total_days, overlap_pct=0.5):
        windows = []
        step_size = int(self.test_period_days * (1 - overlap_pct))

        current_test_start = self.min_training_days
        while current_test_start + self.test_period_days + self.forecast_horizon <=total_days:
            train_start = 0
            train_end = current_test_start - 1
            test_start = current_test_start
            test_end = current_test_start + self.test_period_days - 1

            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'forecast_date': test_end + self.forecast_horizon
            })
            current_test_start +=step_size
        return windows
    def run_backtest(self,symbol, historical_data, model_function, overlap_pct=0.5):
        total_days = len(historical_data)
        windows = self.generate_test_windows(total_days, overlap_pct)

        print(f"\n{'='*80}")
        print(f"Backtesting {symbol}")
        print(f"{'='*80}")
        print(f"Total data points: {total_days} days")
        print(f"Test period length: {self.test_period_days} days (~{self.test_period_days/21:.1f} months)")
        print(f"Number of test windows: {len(windows)}")
        print(f"Forecast horizon: {self.forecast_horizon} days")
        print(f"{'='*80}\n")

        results = []

        for i, window in enumerate(windows):
            try:
                print(f"Testing Window {i+1}/{len(windows)}", end='')

                train_data = historical_data.iloc[window['train_start']:window['train_end']+1]

                test_data = historical_data.iloc[window['test_start']:window['test_end']+1]
                test_prices = test_data['Close'].values

                realized_metrics = self.calculate_realized_risk_metrics(test_prices)
                actual_forecast_price = historical_data.iloc[window['forecast_date']]['Close']
                test_start_price = historical_data.iloc[window['test_start']]['Close']

                predictions = model_function(historical_data, window['train_end'])
                
                actual_returns = (test_data['Close'].values - test_start_price) / test_start_price * 100
                actual_final_return = (actual_forecast_price - test_start_price) / test_start_price * 100

                result = {
                    'window': i + 1,
                    'test_start_date': historical_data.iloc[window['test_start']]['Date'],
                    'test_end_date': historical_data.iloc[window['test_end']]['Date'],
                    'forecast_date': historical_data.iloc[window['forecast_date']]['Date'],
                    'start_price': test_start_price,
                    
                    'actual_forecast_price': actual_forecast_price,
                    'predicted_price': predictions.get('predicted_price', np.nan),
                    'actual_return_pct': ((actual_forecast_price - test_start_price) / test_start_price) * 100,
                    'predicted_return_pct': predictions.get('predicted_return_pct', np.nan),
                    
                    'realized_sharpe': realized_metrics['sharpe_ratio'],
                    'realized_sortino': realized_metrics['sortino_ratio'],
                    'realized_max_drawdown': realized_metrics['max_drawdown_pct'],
                    'realized_volatility': realized_metrics['annualized_volatility'],
                    'realized_total_return': realized_metrics['total_return_pct'],
                    
                    'predicted_sharpe': predictions.get('sharpe_ratio', np.nan),
                    'predicted_sortino': predictions.get('sortino_ratio', np.nan),
                    'predicted_max_drawdown': predictions.get('max_drawdown', np.nan),
                    'predicted_volatility': predictions.get('volatility', np.nan),
                    
                    'predicted_var_5': predictions.get('var_5', np.nan),
                    'var_5_breached': actual_final_return < predictions.get('var_5', 0),
                    'predicted_var_1': predictions.get('var_1', np.nan),
                    'var_1_breached': actual_final_return < predictions.get('var_1', 0),
                    
                    'predicted_prob_profit': predictions.get('prob_profit', np.nan),
                    'actual_profit': realized_metrics['total_return_pct'] > 0,
                }
                
                results.append(result)
                print("Backtest Worked")
                
            except Exception as e:
                print(f"Error running backtest: {e}")
                continue
        
        self.results = pd.DataFrame(results)
        return self.results

    def calculate_validation_metrics(self):
        if len(self.results) == 0:
            print("No results to validate")
            return 0
        df = self.results

        price_errors = df['predicted_price'] - df['actual_forecast_price']
        price_pct_error = (price_errors / df['actual_forecast_price']) * 100
        return_errors = df['predicted_return_pct'] - df['actual_return_pct']

        sharpe_errors = df['predicted_sharpe'] - df['realized_sharpe']
        sharpe_mae = np.abs(sharpe_errors).mean()
        sharpe_rmse = np.sqrt((sharpe_errors**2).mean())
        sharpe_correlation = df[['predicted_sharpe', 'realized_sharpe']].corr().iloc[0,1]

        sortino_errors = df['predicted_sortino'] - df['realized_sortino']
        sortino_mae = np.abs(sortino_errors).mean()
        sortino_rmse = np.sqrt((sortino_errors**2).mean())
        sortino_correlation = df[['predicted_sortino', 'realized_sortino']].corr().iloc[0,1]

        dd_errors = df['predicted_max_drawdown'] - df['realized_max_drawdown']
        dd_mae = np.abs(dd_errors).mean()
        dd_rmse = np.sqrt((dd_errors**2).mean())
        dd_correlation = df[['predicted_max_drawdown', 'realized_max_drawdown']].corr().iloc[0, 1]
        
        # get as close to 1 as possible: vol_ratio_mean
        vol_ratio = df['predicted_volatility'] / df['realized_volatility']
        vol_ratio_mean = vol_ratio.mean()
        vol_ratio_std = vol_ratio.std()
        vol_correlation = df[['predicted_volatility', 'realized_volatility']].corr().iloc[0, 1]
        
        var_5_breach_rate = df['var_5_breached'].mean()
        var_1_breach_rate = df['var_1_breached'].mean()
        
        predicted_direction = np.sign(df['predicted_return_pct'])
        actual_direction = np.sign(df['actual_return_pct'])
        directional_accuracy = (predicted_direction == actual_direction).mean()

        metrics = {
            'n_tests': len(df),
            'test_period_days': self.test_period_days,
            'forecast_horizon': self.forecast_horizon,
            
            'price_mae': np.abs(price_errors).mean(),
            'price_rmse': np.sqrt((price_errors**2).mean()),
            'return_mae': np.abs(return_errors).mean(),
            'directional_accuracy': directional_accuracy,
            
            # === RISK METRIC PREDICTIONS (MOST IMPORTANT) ===
            'sharpe_ratio': {
                'mae': sharpe_mae,
                'rmse': sharpe_rmse,
                'correlation': sharpe_correlation,
                'bias': sharpe_errors.mean(),
            },
            'sortino_ratio': {
                'mae': sortino_mae,
                'rmse': sortino_rmse,
                'correlation': sortino_correlation,
                'bias': sortino_errors.mean(),
            },
            'max_drawdown': {
                'mae_pct': dd_mae,
                'rmse_pct': dd_rmse,
                'correlation': dd_correlation,
                'bias_pct': dd_errors.mean(),
            },
            'volatility': {
                'ratio_mean': vol_ratio_mean,  # close to 1 as possible
                'ratio_std': vol_ratio_std,
                'correlation': vol_correlation,
                'pass': 0.9 <= vol_ratio_mean <= 1.1,  # Within 10% is good
            },
            
            'var_5_breach_rate': var_5_breach_rate,
            'var_5_pass': 0.03 <= var_5_breach_rate <= 0.07,
            'var_1_breach_rate': var_1_breach_rate,
            'var_1_pass': 0.005 <= var_1_breach_rate <= 0.02,
        }
        
        return metrics
    
    def print_validation_report(self, metrics):
        if metrics is None:
            return
        
        print(f"\n{'='*80}")
        print(f"Backtesting Metrics Report")
        print(f"{'='*80}")
        print(f"Number of test periods: {metrics['n_tests']}")
        print(f"Test period length: {metrics['test_period_days']} days / ~{metrics['test_period_days']/21:.1f} months")
        print(f"Forecast horizon: {metrics['forecast_horizon']} days")
        
        print(f"\n{'-'*80}")
        print("Sharpe Ratio Accuracy")
        print(f"{'-'*80}")
        print(f"Mean Absolute Error:                {metrics['sharpe_ratio']['mae']:.3f}")
        print(f"Root Mean Square Error:             {metrics['sharpe_ratio']['rmse']:.3f}")
        print(f"Correlation (pred vs actual):       {metrics['sharpe_ratio']['correlation']:.3f}")
        print(f"Bias (systematic over/under):       {metrics['sharpe_ratio']['bias']:+.3f}")
        
        print(f"\n{'-'*80}")
        print("Sortino Ratio Accuracy")
        print(f"{'-'*80}")
        print(f"Mean Absolute Error:            {metrics['sortino_ratio']['mae']:.3f}")
        print(f"Root Mean Square Error:         {metrics['sortino_ratio']['rmse']:.3f}")
        print(f"Correlation (pred vs actual):   {metrics['sortino_ratio']['correlation']:.3f}")
        print(f"Bias (systematic over/under):   {metrics['sortino_ratio']['bias']:+.3f}")
        
        print(f"\n{'-'*80}")
        print("Maximum Drawdown Accuracy")
        print(f"{'-'*80}")
        print(f"Mean Absolute Error:            {metrics['max_drawdown']['mae_pct']:.2f}%")
        print(f"Root Mean Square Error:         {metrics['max_drawdown']['rmse_pct']:.2f}%")
        print(f"Correlation (pred vs actual):   {metrics['max_drawdown']['correlation']:.3f}")
        print(f"Bias (systematic over/under):   {metrics['max_drawdown']['bias_pct']:+.2f}%")
        
        print(f"\n{'-'*80}")
        print("Volatility Prediction")
        print(f"{'-'*80}")
        print(f"Predicted/Realized Vol Ratio (mean): {metrics['volatility']['ratio_mean']:.3f}")
        print(f"Predicted/Realized Vol Ratio (std):  {metrics['volatility']['ratio_std']:.3f}")
        print(f"Correlation (pred vs actual):        {metrics['volatility']['correlation']:.3f}")
        print(f"Calibration Test:                    {'✓ PASS' if metrics['volatility']['pass'] else '✗ FAIL'}")
        print(f"Close to 1 as possible")
        print(f"  → Ratio = {metrics['volatility']['ratio_mean']:.3f} means predictions are", end=" ")

        print(f"\n{'-'*80}")
        print("Var Testing")
        print(f"{'-'*80}")
        print(f"VaR 5% Breach Rate:                  {metrics['var_5_breach_rate']:.1%} (target: 5%)")
        print(f"VaR 5% Test Result:                  {'Passed' if metrics['var_5_pass'] else 'Failed'}")
        print(f"VaR 1% Breach Rate:                  {metrics['var_1_breach_rate']:.1%} (target: 1%)")
        print(f"VaR 1% Test Result:                  {'Passed' if metrics['var_1_pass'] else 'Failed'}")
        
        print(f"\n{'='*80}")
        print("Overall Model Rating")
        print(f"{'='*80}")
        
        # Calculate overall score
        binary_scores = []
        binary_scores.append(1.0 if metrics['volatility']['pass'] else 0.0)
        binary_scores.append(1.0 if metrics['var_5_pass'] else 0.0)
        binary_scores.append(1.0 if abs(metrics['sharpe_ratio']['correlation']) > 0.2 else 0.0)
        binary_scores.append(1.0 if abs(metrics['max_drawdown']['correlation']) > 0.3 else 0.0)
        
        binary_score = np.mean(binary_scores) * 100

        if metrics['volatility']['pass'] and abs(metrics['volatility']['correlation']) > 0.5:
            vol_score = 100
        elif metrics['volatility']['pass']:
            vol_score = 85
        elif 0.85 <= metrics['volatility']['ratio_mean'] <= 1.15:
            vol_score = 70
        else:
            vol_score = 40

        var_score = 0
        if metrics['var_5_pass']:
            var_score += 50
        if metrics['var_1_pass']:
            var_score += 50

        sharpe_corr = metrics['sharpe_ratio']['correlation']
        if sharpe_corr > 0.4:
            sharpe_score = 100
        elif sharpe_corr > 0.2:
            sharpe_score = 80
        elif sharpe_corr > 0.1:
            sharpe_score = 65
        elif sharpe_corr > 0:
            sharpe_score = 50
        elif sharpe_corr > -0.1:
            sharpe_score = 30
        else:
            sharpe_score = 0

        dd_bias = abs(metrics['max_drawdown']['bias_pct'])
        dd_corr = metrics['max_drawdown']['correlation']
        
        if dd_bias < 15 and dd_corr > 0.3:
            dd_score = 100
        elif dd_bias < 30 and dd_corr > 0.2:
            dd_score = 85
        elif dd_bias < 50 and dd_corr > 0:
            dd_score = 70
        elif dd_bias < 70:
            dd_score = 50
        else:
            dd_score = 25
        
        weighted_score = (
            vol_score * 0.35 +
            var_score * 0.35 +
            sharpe_score * 0.15 + 
            dd_score * 0.15
        )
        print(f"Binary Score: {binary_score:.0f}/100")
        print(f" • Volatility Calibrated:          {'Yes' if binary_scores[0] else 'No'}")
        print(f" • VaR Tests Passed:                {'Yes' if binary_scores[1] else 'No'}")
        print(f" • Sharpe Correlation > 0.2:        {'Yes' if binary_scores[2] else 'No'}")
        print(f" • Max DD Correlation > 0.3:        {'Yes' if binary_scores[3] else 'No'}")
        
        print(f"Weighted Score: {weighted_score:.0f}/100")
        print(f"  • Volatility:      {vol_score:.0f}/100 (35% weight)")
        print(f"  • VaR Testing:     {var_score:.0f}/100 (35% weight)")
        print(f"  • Sharpe Ratio:    {sharpe_score:.0f}/100 (15% weight)")
        print(f"  • Max Drawdown:    {dd_score:.0f}/100 (15% weight)")
        
        final_score = weighted_score
        
        print(f"\n{'='*80}")
        print(f"FINAL MODEL RATING: {final_score:.0f}/100")
        if final_score >= 85:
            rating = "Elite"
        elif final_score >= 70:
            rating = "Excellent - Strong Performance"
        elif final_score >= 55:
            rating = "Good - Acceptable for Use"
        elif final_score >= 40:
            rating = "Decent - Needs Improvement"
        else:
            rating = "Poor - Significant Issues"
        
        print(f"{rating}")


    def plot_backtest_results(self):
 
        if len(self.results) == 0:
            print("No results to plot!")
            return
        
        df = self.results
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        ax = axes[0, 0]
        ax.scatter(df['realized_sharpe'], df['predicted_sharpe'], alpha=0.6, s=50)
        min_val = min(df['realized_sharpe'].min(), df['predicted_sharpe'].min())
        max_val = max(df['realized_sharpe'].max(), df['predicted_sharpe'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        corr = df[['realized_sharpe', 'predicted_sharpe']].corr().iloc[0, 1]
        ax.set_xlabel('Realized Sharpe Ratio')
        ax.set_ylabel('Predicted Sharpe Ratio')
        ax.set_title(f'Sharpe Ratio Predictions (r={corr:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.scatter(df['realized_sortino'], df['predicted_sortino'], alpha=0.6, s=50)
        min_val = min(df['realized_sortino'].min(), df['predicted_sortino'].min())
        max_val = max(df['realized_sortino'].max(), df['predicted_sortino'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        corr = df[['realized_sortino', 'predicted_sortino']].corr().iloc[0, 1]
        ax.set_xlabel('Realized Sortino Ratio')
        ax.set_ylabel('Predicted Sortino Ratio')
        ax.set_title(f'Sortino Ratio Predictions (r={corr:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        ax.scatter(df['realized_max_drawdown'], df['predicted_max_drawdown'], alpha=0.6, s=50)
        min_val = min(df['realized_max_drawdown'].min(), df['predicted_max_drawdown'].min())
        max_val = max(df['realized_max_drawdown'].max(), df['predicted_max_drawdown'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        corr = df[['realized_max_drawdown', 'predicted_max_drawdown']].corr().iloc[0, 1]
        ax.set_xlabel('Realized Max Drawdown (%)')
        ax.set_ylabel('Predicted Max Drawdown (%)')
        ax.set_title(f'Max Drawdown Predictions (r={corr:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        vol_ratio = df['predicted_volatility'] / df['realized_volatility']
        ax.hist(vol_ratio, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        ax.axvline(1.0, color='r', linestyle='--', linewidth=2, label='Perfect calibration (ratio=1.0)')
        ax.axvline(vol_ratio.mean(), color='g', linestyle='-', linewidth=2, 
                   label=f'Mean: {vol_ratio.mean():.3f}')
        ax.axvspan(0.9, 1.1, alpha=0.2, color='green', label='Acceptable range')
        ax.set_xlabel('Predicted / Realized Volatility')
        ax.set_ylabel('Frequency')
        ax.set_title('Volatility Calibration ("Beta Close to 1")')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        var_5_breaches = df['var_5_breached'].astype(int)
        cumulative_breaches = np.cumsum(var_5_breaches)
        expected_breaches = np.arange(1, len(df) + 1) * 0.05
        
        ax.plot(df['window'], cumulative_breaches, label='Actual VaR 5% breaches', marker='o')
        ax.plot(df['window'], expected_breaches, 'r--', label='Expected (5% rate)')
        ax.fill_between(df['window'], expected_breaches * 0.6, expected_breaches * 1.4, 
                        alpha=0.2, color='green', label='Acceptable range')
        ax.set_xlabel('Test Window')
        ax.set_ylabel('Cumulative Breaches')
        ax.set_title('VaR 5% Backtesting')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 2]
        sharpe_errors = df['predicted_sharpe'] - df['realized_sharpe']
        sortino_errors = df['predicted_sortino'] - df['realized_sortino']
        
        ax.plot(df['window'], sharpe_errors, marker='o', label='Sharpe Error', alpha=0.7)
        ax.plot(df['window'], sortino_errors, marker='s', label='Sortino Error', alpha=0.7)
        ax.axhline(0, color='r', linestyle='--', linewidth=1)
        ax.axhline(sharpe_errors.mean(), color='blue', linestyle=':', 
                  label=f'Sharpe bias: {sharpe_errors.mean():+.2f}')
        ax.axhline(sortino_errors.mean(), color='orange', linestyle=':', 
                  label=f'Sortino bias: {sortino_errors.mean():+.2f}')
        ax.set_xlabel('Test Window')
        ax.set_ylabel('Prediction Error')
        ax.set_title('Risk Metric Prediction Errors Over Time')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()