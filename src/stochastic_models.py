#Imports
import numpy as np
from arch import arch_model
import pandas as pd




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

def garch_volatility_forecast(log_returns, forecast_days=30,verbose = False):
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

        if verbose:
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
        if verbose:
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
            
            last_return = returns[-1]
            last_var = cond_var[-1]
            
            forecasted_vars = np.zeros(forecast_days)
            forecasted_vars[0] = omega + alpha * last_return**2 + beta * last_var
            
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
            if verbose:
                print(f"GARCH estimation failed: {e}")
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
        return garch_volatility_forecast(log_returns, forecast_days=forecast_days)
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


    


def stochastic_analysis(analysis_result, n_simulations=1000, forecast_days = 30, compare_models=False, verbose = False):
    try:
        closing_prices = analysis_result['closing_prices']
        symbol = analysis_result['symbol']
        current_price = analysis_result['current_price']
        if verbose:
            print(f"\nRunning Stochastic Analysis for {symbol}....")

        gbm_params = estimate_gbm_params(closing_prices)
        if verbose:
            print(f"GBM Parameters:")
            print(f"  Daily drift (μ): {gbm_params['mu_daily']:.4f}")
            print(f"  Daily volatility (σ): {gbm_params['sigma_daily']:.4f}")
            print(f"  Annual drift: {gbm_params['mu_annual']:.2%}")
            print(f"  Annual volatility: {gbm_params['sigma_annual']:.2%}")
        
        best_garch = None
        comparison_df = None

        use_garch_for_simulation = forecast_days <= 60


        if use_garch_for_simulation and compare_models:
            result_tuple = compare_garch_models(gbm_params['log_returns'], forecast_days)
            if result_tuple and isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                best_garch, comparison_df = result_tuple
            else:
                best_garch = result_tuple
            garch_result = best_garch
        elif use_garch_for_simulation:
            garch_result = garch_volatility_forecast(gbm_params['log_returns'], forecast_days)
            best_garch = garch_result
        else:
            best_garch=None
            garch_result = None
        if best_garch is None or not isinstance(best_garch, dict):
            if verbose:
                print("Using constant volatility fallback over GARCH")
            constant_vol = gbm_params['sigma_daily']
            best_garch = {
                'forecasted_volatility': [constant_vol] * forecast_days,
                'conditional_variances': None,
                'omega': None,
                'alpha': None,
                'beta': None,
                'avg_forecast_vol': constant_vol
            }
        has_valid_garch = False
        if best_garch is not None and isinstance(best_garch,dict):
            forecasted_vol = best_garch.get('forecasted_volatility')
            if forecasted_vol is not None:
                try:
                    vol_length = len(forecasted_vol)
                    has_valid_garch = vol_length > 0
                except:
                    has_valid_garch = False
        if has_valid_garch:
            simulations = np.zeros((n_simulations, forecast_days + 1))
            simulations[:, 0] = current_price

            forecasted_vols = best_garch['forecasted_volatility']
            if len(forecasted_vols) < forecast_days:
                long_term_vol = gbm_params['sigma_daily']
            else:
                long_term_vol = gbm_params['sigma_daily']
            for t in range(1, forecast_days + 1):
                if t-1 < len(forecasted_vols):
                    vol = forecasted_vols[t-1]
                else:
                    vol = long_term_vol
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
        if verbose:
            print(f"\n30-Day Stochastic Forecast Results:")
            print(f"  Expected price: ${final_stats['mean_price']:.2f}")
            print(f"  Median price: ${final_stats['median_price']:.2f}")
            print(f"  Price range: ${final_stats['min_price']:.2f} - ${final_stats['max_price']:.2f}")
            print(f"  Probability of profit: {final_stats['prob_profit']:.1%}")
            print(f"  Probability of >5% gain: {final_stats['prob_gain_5pct']:.1%}")
            print(f"  Probability of >5% loss: {final_stats['prob_loss_5pct']:.1%}")
            
        result = {
            'simulations': simulations,
            'gbm_params': gbm_params,
            'best_garch': best_garch,
            'forecast_stats': forecast_stats,
            'final_stats': final_stats,
            'forecast_days': forecast_days,
            'n_simulations': n_simulations,
        }
        if comparison_df is not None:
            result['model_comparison'] = comparison_df
    
        return result
    except Exception as e:
        print(f"Error in stochastic analysis: {e}")
        import traceback
        traceback.print_exc()
        return None