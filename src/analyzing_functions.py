#Used for all Equations + their Metrics


from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import sympy as sym
from scipy.interpolate import CubicSpline, BSpline, splrep, splev
from yahooquery import search, Ticker, Screener
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from .plots import print_risk_report, calculate_comprehensive_risk_metrics, plot_fan_chart, plot_stock_analysis, plot_risk_metrics_dashboard



def enhanced_stock_analysis(symbol, user_prompt=None, company_name=None):
    from .stochastic_models import stochastic_analysis
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
    from .sentiment_finder import get_stock_sentiment
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
                    if np.sign(first_deriv_values[i-1]) != np.sign(first_deriv_values[i]):
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
                    if np.sign(first_deriv_values[i-1]) != np.sign(first_deriv_values[i]):
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