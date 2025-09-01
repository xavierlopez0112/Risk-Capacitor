auto_adjust takes dividends and splits into account. Without it set to true, the price would be shown as the historical prices that they were traded as on the day. With it on, they are adjusted for what corporate actions are taken.
Line 32 makes an array of values after taking original data from the closing price and removing any Na and then showing the values.
Uses generative ai to give the top predictions then takes those predictions and makes equations and charts.
Uses NEWSAPI and VaderSentiments to look through articles. Then tags certain articles and has OpenAI review it which is better than Vader. Sees which top 10 have lowest MSE and positive sentiment and reports the graphs and equations and predictions for those.
.json takes JSON style string and converts it into python dictionary