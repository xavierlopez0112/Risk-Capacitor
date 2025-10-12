# Risk-Capacitor
(Name inspired by Back To The Future's Flux Capacitor)

**Overview**
This project demonstrates risk analysis for stocks. It focuses on volatility metrics such as Value at Risk, Sharpe/Sortino Ratios and Drawdowns instead of focusing on forecasting prices for stocks.
I combined my skills and knowledge in math, coding and finance in the making of this project.

**Motivation**
After taking AP CSA and AP Calc AB in my junior year, I continued my Math/CS progression and I am currently taking AP Stats and AP Calc BC in my senior year. In March-May of my Junior year, I struggled with deciding on what to major in during college. For a while, I thought my major would be Computer Science, but after taking Calculus, I realized that the functional applications of math in the world interested me way more than computer science. That's when I decided to build a project to see if 
#1, I really enjoyed math and using it in combination with coding & #2 if I was any good at it.

**Core Features**
1. Regression and Forecasting Models
        Polynomial Fitting (Degrees: 1-4)
        Cubic Spline and B Spline Interpolation
2. Volatility Analysis
        EWMA (Exponentially Weighted Moving Average) - Estimates Volatility & gives more weight to the more recent data
        Historical Volatility - Calculated with Lookback windows
        GBM (Geometric Browniam Motion) - Stochastic Calculus Equation to estimate volatility
3. Risk Metrics
        VaR (Value at Risk) - Estimated maximum potential loss for 1%, 5% and 10% confidence levels
        CVaR (Conditional Value at Risk) - How much is really lost if VaR threshold is exceeded
        Sharpe Ratio - Indicates level of return per risk taken (Ratio: 1.0-1.9 is good; 2.0-2.9 is very good; 3.0+ is excellent)
        Sortino Ratio - Indicates level of return per downside risk taken (Ratio: 1.0-1.9 is good; 2.0-2.9 is very good; 3.0+ is excellent)
        Maximum Drawdown - Finds a stock's historical volatility and potential for loss (Lower value means less risk)
4. Stock Sentiment
        Uses RSS and News APIs to find the sentiment of a stock
        What I used: Yahoo Finance RSS, Google News RSS, Finnhub API, FinBERT Sentiment Analysis
5. Visualization
        Used matplotlib to show: Fan Charts of Probability, Distribution plots of prices, Monte Carlo Paths, All Risk Metrics

**How it was Built**
Coded in VS Code using Python3. Used libraries for data analysis, modeling and visualization.
Main Libraries:
    numpy & pandas – Data manipulation and basic computations
    matplotlib – Visualization of stock prices and risk metrics
    yfinance – Downloading historical stock price data
    scipy – Statistical calculations for bootstrapping and risk metrics

Project Structure:
All code is in the 'src' folder
    analyzing_functions.py – Core risk analysis and bootstrap simulations
    plots.py – Functions to generate risk and price visualizations
    model_wrapper.py – Wraps models for prediction and backtesting
    backtesting.py – Backtesting framework
    stochastic_models.py – Stochastic simulations for risk metrics
    sentiment_finder.py – Finds Stock's overall sentiment
    main.py – Where the full analysis is ran

Modeling and Analysis:
    Volatility: EWMA combined with Historical Volatility and GBM
    Value at Risk: Bootstrap simulations with 1000 simulated price paths
    Sharpe/Sortino Ratios: Momentum-Based using a 63-day lookback period (~1 quarter)
    Max Drawdown: 25th percentile of bootstrap simulations
    Backtesting: Rolling windows with overlapping periods to evaluate model's predictive performance

Workflow:
1. Download historical stock data using yfinance
2. Clean and structure the data for modeling
3. Run backtests across multiple rolling windows
4. Compute and validate risk metrics (volatility, VaR, max drawdown)
5. Visualize predicted vs. realized metrics
6. Save results for further analysis and reporting



**What I would have done differently & What I learned**
Something I would have done differently is building more and planning less. Even though I started planning and thinking about how to build this in May, I think jumping in and starting to build earlier would have helped me to get the program built faster and given me more ideas on what to do as I built.
I learned a lot building this such as equations from Stochastic Calculus, Basic and some Advanced Statistics, APIs, Python Libraries and how to use Github.

**Resources Used**
- [ChatGPT](https://chatgpt.com/) - Used to help with README, Explanations of Financial Terms, Overall Structure of project and how to use Github.
- [Claude AI](https://claude.ai/) - Helped to debug code, fix errors and brainstorm coding structure. All code was implemented, tested and verified by me.
- [Spline Interpolation](https://www.youtube.com/watch?v=9R5UxtJQNNg) - Regression Models
- [GBM](https://www.youtube.com/watch?v=R8pDHvPWtE4) - Stochastic and Volatility Analysis
- [Finnhub News](https://finnhub.io/) - News
- [YFinance Screeners](https://finance.yahoo.com/research-hub/screener) - Used for inputting stocks
- [Finbert Sentiment Analysis](https://huggingface.co/ProsusAI/finbert?text=Stocks+rallied+and+the+British+pound+gained.) - Analyze News Sentiment
- [Data Analysis Book](Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython by Wes McKinney)



