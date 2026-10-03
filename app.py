
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

st.set_page_config(page_title="Market Trend Scanner", page_icon="📈", layout="wide")

st.title("📈 Market Trend Scanner")
st.caption("Multi-stock research scanner using price, volume, momentum, volatility and market-relative data.")

# ---------------- UNIVERSES ----------------

UK = {
    "HSBA.L":"HSBC", "SHEL.L":"Shell", "BP.L":"BP", "AZN.L":"AstraZeneca",
    "GSK.L":"GSK", "ULVR.L":"Unilever", "RIO.L":"Rio Tinto",
    "LSEG.L":"London Stock Exchange Group", "REL.L":"RELX",
    "DGE.L":"Diageo", "BATS.L":"British American Tobacco",
    "BARC.L":"Barclays", "LLOY.L":"Lloyds", "NWG.L":"NatWest",
    "VOD.L":"Vodafone", "NG.L":"National Grid", "BA.L":"BAE Systems",
    "RR.L":"Rolls-Royce", "IMB.L":"Imperial Brands", "PRU.L":"Prudential",
    "GLEN.L":"Glencore", "AAL.L":"Anglo American", "CNA.L":"Centrica",
    "STAN.L":"Standard Chartered", "TSCO.L":"Tesco"
}

US = {
    "AAPL":"Apple", "MSFT":"Microsoft", "NVDA":"NVIDIA", "AMZN":"Amazon",
    "GOOGL":"Alphabet", "META":"Meta", "TSLA":"Tesla", "AVGO":"Broadcom",
    "JPM":"JPMorgan", "V":"Visa", "MA":"Mastercard", "WMT":"Walmart",
    "COST":"Costco", "NFLX":"Netflix", "AMD":"AMD", "ORCL":"Oracle",
    "CRM":"Salesforce", "ADBE":"Adobe", "INTC":"Intel", "QCOM":"Qualcomm",
    "MU":"Micron", "PLTR":"Palantir", "UBER":"Uber", "DIS":"Disney",
    "KO":"Coca-Cola", "PEP":"PepsiCo", "MCD":"McDonald's", "NKE":"Nike",
    "XOM":"Exxon Mobil", "CVX":"Chevron"
}

BENCHMARKS = {
    "UK Market": "^FTSE",
    "US Market": "^GSPC"
}

FEATURES = [
    "ret1","ret5","ret10","ret20","ret60",
    "ma10dist","ma20dist","ma50dist","ma100dist","ma200dist",
    "ma20_50","ma50_200","RSI","MACD","MACD_signal","MACD_hist",
    "ATR_pct","vol_ratio","vol_trend","hi20dist","lo20dist",
    "hi60dist","lo60dist","bb_pos","relative20","relative60"
]

@st.cache_data(ttl=3600, show_spinner=False)
def get_prices(ticker, years):
    x = yf.download(ticker, period=f"{years}y", interval="1d",
                    auto_adjust=True, progress=False)
    if x.empty:
        return pd.DataFrame()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in x]
    return x[cols].dropna()

def add_features(df, benchmark=None):
    x = df.copy()
    c,h,l,v = x["Close"],x["High"],x["Low"],x["Volume"]

    for p in [10,20,50,100,200]:
        x[f"MA{p}"] = c.rolling(p).mean()
        x[f"ma{p}dist"] = c/x[f"MA{p}"]-1

    x["ret1"]=c.pct_change()
    x["ret5"]=c.pct_change(5)
    x["ret10"]=c.pct_change(10)
    x["ret20"]=c.pct_change(20)
    x["ret60"]=c.pct_change(60)
    x["ma20_50"]=x["MA20"]/x["MA50"]-1
    x["ma50_200"]=x["MA50"]/x["MA200"]-1

    d=c.diff()
    g=d.clip(lower=0).rolling(14).mean()
    loss=(-d.clip(upper=0)).rolling(14).mean()
    rs=g/loss.replace(0,np.nan)
    x["RSI"]=100-100/(1+rs)

    e12=c.ewm(span=12,adjust=False).mean()
    e26=c.ewm(span=26,adjust=False).mean()
    x["MACD"]=e12-e26
    x["MACD_signal"]=x["MACD"].ewm(span=9,adjust=False).mean()
    x["MACD_hist"]=x["MACD"]-x["MACD_signal"]

    prev=c.shift(1)
    tr=pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    x["ATR_pct"]=tr.rolling(14).mean()/c

    v20=v.rolling(20).mean()
    v60=v.rolling(60).mean()
    x["vol_ratio"]=v/v20
    x["vol_trend"]=v20/v60-1

    for p in [20,60]:
        hi=h.rolling(p).max().shift(1)
        lo=l.rolling(p).min().shift(1)
        x[f"hi{p}dist"]=c/hi-1
        x[f"lo{p}dist"]=c/lo-1

    mid=c.rolling(20).mean()
    sd=c.rolling(20).std()
    x["bb_pos"]=(c-(mid-2*sd))/((mid+2*sd)-(mid-2*sd))

    if benchmark is not None and not benchmark.empty:
        b=benchmark["Close"].reindex(x.index).ffill()
        x["relative20"]=(c.pct_change(20)-b.pct_change(20))
        x["relative60"]=(c.pct_change(60)-b.pct_change(60))
    else:
        x["relative20"]=0
        x["relative60"]=0

    return x

def train_and_test(df,horizon):
    x=df.copy()
    x["future_return"]=x["Close"].shift(-horizon)/x["Close"]-1
    x["target"]=(x["future_return"]>0).astype(int)
    x=x.dropna(subset=FEATURES+["future_return"])
    if len(x)<500:
        return None

    X=x[FEATURES]; y=x["target"]
    splitter=TimeSeriesSplit(n_splits=5)
    preds=[]; actual=[]
    for tr,te in splitter.split(X):
        model=RandomForestClassifier(
            n_estimators=250,max_depth=8,min_samples_leaf=10,
            max_features="sqrt",class_weight="balanced",
            random_state=42,n_jobs=-1
        )
        model.fit(X.iloc[tr],y.iloc[tr])
        preds.extend(model.predict(X.iloc[te]))
        actual.extend(y.iloc[te])

    accuracy=accuracy_score(actual,preds)

    final=RandomForestClassifier(
        n_estimators=400,max_depth=8,min_samples_leaf=10,
        max_features="sqrt",class_weight="balanced",
        random_state=42,n_jobs=-1
    )
    final.fit(X,y)

    latest=X.iloc[-1].values.reshape(1,-1)
    p=final.predict_proba(latest)[0,1]
    signal="UPWARD" if p>=0.60 else ("DOWNWARD" if p<=0.40 else "NEUTRAL")

    return {
        "signal":signal,"prob":p,"accuracy":accuracy,
        "price":x["Close"].iloc[-1],
        "volume":x["vol_ratio"].iloc[-1],
        "rsi":x["RSI"].iloc[-1],
        "ret20":x["ret20"].iloc[-1],
        "ret60":x["ret60"].iloc[-1],
        "data":x,"model":final
    }

def scan(universe, years, horizons):
    rows=[]
    details={}
    progress=st.progress(0)
    tickers=list(universe)
    for i,t in enumerate(tickers):
        try:
            raw=get_prices(t,years)
            if raw.empty:
                continue
            market="^FTSE" if t.endswith(".L") else "^GSPC"
            bench=get_prices(market,years)
            data=add_features(raw,bench)
            details[t]=data
            for name,days in horizons.items():
                r=train_and_test(data,days)
                if r:
                    rows.append({
                        "Ticker":t,"Company":universe[t],"Horizon":name,
                        "Signal":r["signal"],"Probability Up":r["prob"],
                        "Price":r["price"],"Volume Ratio":r["volume"],
                        "RSI":r["rsi"],"20D Return":r["ret20"],
                        "60D Return":r["ret60"],
                        "Backtest Accuracy":r["accuracy"]
                    })
        except Exception as e:
            pass
        progress.progress((i+1)/len(tickers))
    progress.empty()
    return pd.DataFrame(rows),details

with st.sidebar:
    st.header("Scanner")
    market_choice=st.selectbox("Market",["UK","US","UK + US","Custom"])
    years=st.slider("Historical years",3,15,8)
    horizons_selected=st.multiselect("Horizons",["5D","10D","20D","60D"],
                                     ["5D","20D","60D"])

    if market_choice=="UK":
        universe=UK
    elif market_choice=="US":
        universe=US
    elif market_choice=="UK + US":
        universe={**UK,**US}
    else:
        custom=st.text_area("Enter tickers separated by commas",
                            "AAPL,MSFT,NVDA,SHEL.L,BP.L")
        universe={t.strip().upper():t.strip().upper()
                  for t in custom.split(",") if t.strip()}

    st.write(f"**{len(universe)} companies selected**")
    run=st.button("🔎 Scan market",use_container_width=True)

if run:
    horizon_map={"5D":5,"10D":10,"20D":20,"60D":60}
    with st.spinner("Running walk-forward models. This can take a few minutes..."):
        results,details=scan(universe,years,
                             {h:horizon_map[h] for h in horizons_selected})
    st.session_state.results=results
    st.session_state.details=details

if "results" not in st.session_state:
    st.info("Choose UK, US or UK + US, then tap **Scan market**.")
    st.markdown("""
### Scanner features
**Trend:** moving averages, momentum, breakouts and Bollinger position  
**Volume:** current volume versus 20-day average and volume trend  
**Momentum:** RSI, MACD and multi-period returns  
**Risk:** ATR and volatility  
**Relative strength:** performance versus the relevant market index  
**Machine learning:** Random Forest with walk-forward historical testing  
**Horizons:** 5, 10, 20 and 60 trading days
""")
else:
    results=st.session_state.results
    details=st.session_state.details
    if results.empty:
        st.error("No results were produced. Check your tickers or try again.")
    else:
        st.subheader("Market results")
        sort_option=st.selectbox("Sort by",
                                  ["Probability Up","Backtest Accuracy","Volume Ratio","20D Return"])
        table=results.sort_values(sort_option,ascending=False).copy()
        fmt=table.copy()
        for c in ["Probability Up","Backtest Accuracy","20D Return","60D Return"]:
            fmt[c]=fmt[c].map(lambda v:f"{v:.1%}")
        fmt["Volume Ratio"]=fmt["Volume Ratio"].map(lambda v:f"{v:.2f}x")
        fmt["RSI"]=fmt["RSI"].map(lambda v:f"{v:.1f}")
        st.dataframe(fmt,use_container_width=True,hide_index=True)

        st.download_button("⬇️ Download CSV",
                           results.to_csv(index=False).encode(),
                           "market_scan_results.csv","text/csv")

        st.subheader("Company detail")
        ticker=st.selectbox("Select company",sorted(details))
        horizon=st.selectbox("Prediction horizon",horizons_selected)
        r=train_and_test(details[ticker],{"5D":5,"10D":10,"20D":20,"60D":60}[horizon])

        a,b,c,d=st.columns(4)
        a.metric("Signal",r["signal"])
        b.metric("Probability Up",f"{r['prob']:.1%}")
        c.metric("Backtest Accuracy",f"{r['accuracy']:.1%}")
        d.metric("Volume vs 20D",f"{r['volume']:.2f}x")

        fig=go.Figure()
        data=details[ticker]
        fig.add_trace(go.Candlestick(x=data.index,open=data["Open"],high=data["High"],
                                     low=data["Low"],close=data["Close"],name="Price"))
        fig.add_trace(go.Scatter(x=data.index,y=data["MA50"],name="50D MA"))
        fig.add_trace(go.Scatter(x=data.index,y=data["MA200"],name="200D MA"))
        fig.update_layout(height=550,xaxis_rangeslider_visible=False)
        st.plotly_chart(fig,use_container_width=True)

st.caption("For research and educational use. A model probability is not a guarantee of future performance and should not be treated as financial advice.")
