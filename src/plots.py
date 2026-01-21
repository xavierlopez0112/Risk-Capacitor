#Plot imports
import numpy as np
import traceback
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splev

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
    from .stochastic_models import stochastic_analysis

    from .analyzing_functions import analyze_stock
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

        gain_to_loss_ratio = abs(expected_gain_if_positive / expected_loss_if_negative) if expected_loss_if_negative != 0 else float('inf')

        volatility_metrics = {
            'return_std': np.std(returns_pct),
            'return_variance': np.var(returns_pct),
            'annualized_volatility': np.std(returns_pct) * np.sqrt(252/forecast_days),
            'coefficient_of_variation': np.std(returns_pct) / np.mean(returns_pct) if np.mean(returns_pct) != 0 else float('inf')
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