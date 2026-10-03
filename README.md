# Market Trend Scanner — iPad Web App

## What this version adds

- UK market preset
- US market preset
- UK + US combined scan
- Custom ticker list
- 5D / 10D / 20D / 60D horizons
- Walk-forward backtesting
- Random Forest probability model
- Volume analysis
- RSI / MACD
- Moving averages
- Breakout measures
- Volatility / ATR
- Relative strength versus FTSE 100 for UK shares and S&P 500 for US shares
- Interactive charts
- CSV export

## iPad

This is a Streamlit web application. The iPad does not need Python installed.

Recommended deployment:
1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Deploy the repository using Streamlit Community Cloud.
4. Open the resulting web address in Safari on iPad.
5. Use Safari's Share menu -> Add to Home Screen.

## Local testing

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

The built-in stock lists are a practical starter universe, not a complete representation of every listed company. Additional symbols can be entered through Custom.

The scanner is a research tool. Historical backtest accuracy and model probabilities do not guarantee future performance.
