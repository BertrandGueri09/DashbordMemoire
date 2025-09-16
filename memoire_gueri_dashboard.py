# memoire_gueri_dashboard.py — Titre centré propre + Filtre global pour tous les onglets
# ------------------------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import io, os, re, warnings
from typing import Union, Dict, Tuple, List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statsmodels pour prévisions
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# ====================== CONFIG APP ======================
st.set_page_config(
    page_title="Dashboard Marchés Boursiers – BRVM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_SHARES_OUTSTANDING = 181_371_900
DEFAULT_PRICE_PATH = "CFAOCI_filtre.csv"
DEFAULT_DPS_PATH   = "dps_exemple.csv"
DEFAULT_EPS_PATH   = "eps_exemple.csv"
DEFAULT_NET_PATH   = "net_income_exemple.csv"

# ====================== THEME SOMBRE (léger) ======================
BASE_CSS = """
<style>
.block-container {max-width: 1600px; padding-top: 0.6rem;}
[data-testid="stMetricValue"] {font-size: 1.15rem;}
p, li {line-height: 1.35}
</style>
"""
DARK_CSS = """
<style>
body, .block-container { background-color: #0e1117; color: #e8e6e3; }
</style>
"""
def apply_dark_theme():
    st.markdown(BASE_CSS, unsafe_allow_html=True)
    st.markdown(DARK_CSS, unsafe_allow_html=True)

def set_fig_template(fig: go.Figure):
    fig.update_layout(template="plotly_dark",
                      paper_bgcolor="#0e1117",
                      plot_bgcolor="#0e1117",
                      font=dict(color="#e8e6e3"))

# ====================== I/O ======================
@st.cache_data
def load_data(path_or_buffer: Union[str, io.BytesIO]) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Dernier':'Close','Ouv.':'Open','Ouv':'Open',
                            'Plus Haut':'High','Plus Bas':'Low','Vol.':'Volume',
                            'Variation %':'Variation'})
    if 'Date' not in df.columns:
        raise ValueError("Colonne 'Date' introuvable.")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])

    def to_num(x):
        if pd.isna(x): return np.nan
        s = str(x).replace('\u202f','').replace('\xa0','').replace(' ','').replace(',', '.')
        s = re.sub(r'[^0-9.\-]', '', s)
        return pd.to_numeric(s, errors='coerce')

    for c in ['Close','Open','High','Low']:
        if c in df.columns: df[c] = df[c].apply(to_num)

    def parse_vol(v):
        if pd.isna(v) or v=='': return 0.0
        s = str(v).strip().replace('\u202f','').replace('\xa0','').replace(' ','').replace(',', '.')
        m = re.match(r'^(-?\d+(\.\d+)?)([kKmM]?)$', s)
        if not m:
            s = re.sub(r'[^0-9.\-]','', s); return float(s) if s else 0.0
        val = float(m.group(1)); suf = m.group(3).lower()
        if suf=='k': val *= 1_000
        if suf=='m': val *= 1_000_000
        return val
    df['Volume'] = df['Volume'].apply(parse_vol) if 'Volume' in df.columns else 0.0
    if 'Variation' in df.columns: df['Variation'] = df['Variation'].apply(to_num)

    need = ['Date','Close','Open','High','Low']
    df = df.dropna(subset=need).sort_values('Date').reset_index(drop=True)
    df = df[df['High'] >= df['Low']]
    df = df[(df['High'] >= df['Open']) & (df['High'] >= df['Close'])]
    df = df[(df['Low']  <= df['Open']) & (df['Low']  <= df['Close'])]
    return df

def resample_ohlcv(df: pd.DataFrame, freq_code: str) -> pd.DataFrame:
    dfi = df.set_index('Date')
    out = dfi.resample(freq_code).agg({
        'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum','Variation':'mean'
    }).dropna(subset=['Open','High','Low','Close']).reset_index()
    return out

# ====================== INDICATEURS ======================
def sma(s, w): return s.rolling(w, min_periods=1).mean()
def ema(s, w): return s.ewm(span=w, adjust=False, min_periods=1).mean()

def rsi(prices: pd.Series, window=14):
    d = prices.diff(); up, dn = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rd = dn.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = ru / rd.replace(0, np.nan)
    return (100 - (100/(1+rs))).clip(0,100).fillna(50)

def bb(prices, w=20, n=2.0):
    m = prices.rolling(w, min_periods=1).mean()
    s = prices.rolling(w, min_periods=1).std(ddof=0)
    return m - n*s, m, m + n*s

def macd(prices, f=12, s=26, sig=9):
    ef, es = ema(prices, f), ema(prices, s)
    line = ef - es
    signal = line.ewm(span=sig, adjust=False, min_periods=1).mean()
    hist = line - signal
    return line, signal, hist

def _ann_factor(code: str) -> float:
    return {'D':252.0, 'W':52.0, 'M':12.0}.get(code, 252.0)

def perf_metrics(df, rf_annual_pct=0.0, freq_code='D'):
    latest, oldest = df.iloc[-1], df.iloc[0]
    total_return = (latest['Close']/oldest['Close'] - 1)*100
    ret = df['Close'].pct_change().dropna()
    ann = _ann_factor(freq_code)
    if ret.empty:
        return dict(current_price=latest['Close'], total_return=0, annualized_return=0, volatility=0,
                    sharpe=0, max_drawdown=0, avg_volume=df['Volume'].mean(),
                    max_price=df['Close'].max(), min_price=df['Close'].min(),
                    last_update=latest['Date'].strftime('%d/%m/%Y'))
    m, s = ret.mean(), ret.std()
    ann_return = ((1+m)**ann - 1)*100
    vol = s*np.sqrt(ann)*100
    rf_per = (rf_annual_pct/100)/ann
    sharpe = 0.0 if s==0 else ((m - rf_per)/s)*np.sqrt(ann)
    cum = (1+ret).cumprod()
    mdd = (cum/cum.cummax() - 1).min()*100
    return dict(current_price=latest['Close'], total_return=total_return, annualized_return=ann_return,
                volatility=vol, sharpe=sharpe, max_drawdown=mdd, avg_volume=df['Volume'].mean(),
                max_price=df['Close'].max(), min_price=df['Close'].min(),
                last_update=latest['Date'].strftime('%d/%m/%Y'))

def add_indics(df: pd.DataFrame, p: Dict) -> pd.DataFrame:
    df = df.copy()
    if p.get('show_sma'):
        df['SMA_1'] = sma(df['Close'], p['sma1']); df['SMA_2'] = sma(df['Close'], p['sma2'])
    if p.get('show_ema'): df['EMA_1'] = ema(df['Close'], p['ema1'])
    if p.get('show_bb'):
        lo, mid, up = bb(df['Close'], p['bb_window'], p['bb_std'])
        df['BB_L'], df['BB_M'], df['BB_U'] = lo, mid, up
    if p.get('show_rsi'): df['RSI'] = rsi(df['Close'], p['rsi_window'])
    if p.get('show_macd'):
        L,S,H = macd(df['Close'], p['macd_fast'], p['macd_slow'], p['macd_signal'])
        df['MACD_L'], df['MACD_S'], df['MACD_H'] = L,S,H
    return df

def plot_tech(df, chart_type, p) -> go.Figure:
    rows = 1 + int(p.get('show_rsi')) + int(p.get('show_macd'))
    rh = [1.0] if rows==1 else ([0.68,0.32] if rows==2 else [0.6,0.22,0.18])
    titles = ['Prix & Volume'] + (['RSI'] if p.get('show_rsi') else []) + (['MACD'] if p.get('show_macd') else [])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=rh, subplot_titles=titles)
    if chart_type=='Chandelles':
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='Cours'), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Prix', mode='lines', line=dict(width=2)), row=1, col=1)
    if p.get('show_sma'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_1'], name=f"MM{p['sma1']}", mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_2'], name=f"MM{p['sma2']}", mode='lines'), row=1, col=1)
    if p.get('show_ema'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_1'], name=f"EMA{p['ema1']}", mode='lines'), row=1, col=1)
    if p.get('show_bb'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_M'], name="BB", mode='lines', line=dict(dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_U'], showlegend=False, mode='lines', line=dict(width=0)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_L'], fill='tonexty', mode='lines', line=dict(width=0), name='BB Zone', opacity=0.1), row=1, col=1)
    r = 2
    if p.get('show_rsi'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', mode='lines'), row=r, col=1)
        for y, dash, color in [(70,"dash","red"), (50,"dot","gray"), (30,"dash","green")]:
            fig.add_hline(y=y, line_dash=dash, line_color=color, row=r, col=1)
        r += 1
    if p.get('show_macd'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_L'], name='MACD', mode='lines'), row=r, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_S'], name='Signal', mode='lines'), row=r, col=1)
        fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_H'], name='Hist', opacity=0.6), row=r, col=1)
    fig.update_layout(height=620, hovermode='x unified', showlegend=True,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                      margin=dict(t=30,b=30,l=30,r=20))
    set_fig_template(fig); return fig

# ====================== FONDAMENTAUX (annuels) ======================
def _detect_year_column(df: pd.DataFrame) -> Optional[str]:
    for c in ['Annee','Année','Year','year','period']:
        if c in df.columns: return c
    return None

def compute_fundamentals_from_daily(df_daily: pd.DataFrame, shares_outstanding: int) -> pd.DataFrame:
    if df_daily.empty: return pd.DataFrame()
    df = df_daily.copy().sort_values('Date').set_index('Date')
    df['ret'] = df['Close'].pct_change()
    ann = df.resample('Y').agg(last_price=('Close','last'), avg_price=('Close','mean'), vol_sum=('Volume','sum'))
    ann['last_close_prev'] = ann['last_price'].shift(1)
    ann['annual_return_%'] = ((ann['last_price']/ann['last_close_prev']) - 1.0) * 100

    def annual_vol(g):
        r = g['ret'].dropna()
        return (r.std()*np.sqrt(252)*100) if len(r)>1 else np.nan
    ann['vol_annual_%'] = df.groupby(pd.Grouper(freq='Y')).apply(annual_vol).values

    def max_dd(g):
        c = g['Close'].dropna()
        return (c/c.cummax() - 1.0).min()*100 if not c.empty else np.nan
    ann['max_drawdown_intra_%'] = df.groupby(pd.Grouper(freq='Y')).apply(max_dd).values

    ann['market_cap_fin_annee_FCFA'] = ann['last_price'] * float(shares_outstanding)
    years = ann.index.year.astype(int)
    ann = ann.reset_index(drop=True)
    ann.insert(0, 'Annee', years)
    for c in ['last_price','avg_price','annual_return_%','vol_annual_%','max_drawdown_intra_%']:
        ann[c] = ann[c].astype(float).round(2)
    ann['vol_sum'] = ann['vol_sum'].round(0).astype('Int64')
    ann['market_cap_fin_annee_FCFA'] = ann['market_cap_fin_annee_FCFA'].round(0).astype('Int64')
    return ann

def _parse_year_value_df(uploaded: Union[str, io.BytesIO], value_cols: List[str]) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        return None
    df.columns = df.columns.str.strip()
    y = _detect_year_column(df)
    if not y:
        if 'period' in df.columns: df = df.rename(columns={'period':'Annee'}); y='Annee'
        else: return None
    cand = None
    for c in value_cols:
        if c in df.columns: cand = c; break
        lw = {k.lower():k for k in df.columns}
        if c.lower() in lw: cand = lw[c.lower()]; break
    if cand is None: return None
    out = df[[y, cand]].rename(columns={y:'Annee', cand:cand})
    out['Annee'] = pd.to_numeric(out['Annee'], errors='coerce').astype('Int64')
    out[cand] = pd.to_numeric(out[cand], errors='coerce')
    out = out.dropna(subset=['Annee'])
    return out

def enrich_with_dividends_eps(ann_df: pd.DataFrame, shares_outstanding: int,
                              dps_df: Optional[pd.DataFrame],
                              eps_or_net_df: Optional[pd.DataFrame],
                              manual_dps: Optional[float], manual_payout_pct: Optional[float]) -> pd.DataFrame:
    if ann_df is None or ann_df.empty: return ann_df
    out = ann_df.copy()
    # DPS
    if dps_df is not None and not dps_df.empty:
        if 'DPS' not in dps_df.columns:
            other = [c for c in dps_df.columns if c!='Annee'][0]
            dps_df = dps_df.rename(columns={other:'DPS'})
        out = out.merge(dps_df[['Annee','DPS']], on='Annee', how='left')
    # EPS ou Net Income
    if eps_or_net_df is not None and not eps_or_net_df.empty:
        temp = eps_or_net_df.copy()
        cols = {c.lower(): c for c in temp.columns}
        if 'eps' in cols:
            temp = temp.rename(columns={cols['eps']:'EPS'})
            out = out.merge(temp[['Annee','EPS']], on='Annee', how='left')
        else:
            # net income -> EPS
            for key in ['net_income','resultat_net','rn','benefice','profit']:
                if key in cols:
                    temp = temp.rename(columns={cols[key]:'net_income'})
                    out = out.merge(temp[['Annee','net_income']], on='Annee', how='left')
                    out['EPS'] = out.get('EPS', np.nan)
                    out['EPS'] = out['EPS'].where(out['EPS'].notna(), out['net_income']/float(shares_outstanding))
                    break
    if 'EPS' not in out.columns: out['EPS'] = np.nan

    if manual_dps is not None and manual_payout_pct is not None and len(out)>0:
        try:
            last_year = int(out['Annee'].max())
            payout = max(min(manual_payout_pct/100.0, 0.9999), 0.0001)
            est_eps = manual_dps / payout
            out.loc[out['Annee']==last_year, 'DPS'] = out.loc[out['Annee']==last_year, 'DPS'].fillna(manual_dps)
            out.loc[out['Annee']==last_year, 'EPS'] = out.loc[out['Annee']==last_year, 'EPS'].fillna(est_eps)
        except Exception:
            pass

    if 'DPS' in out.columns:
        out['Dividends_Total_FCFA'] = (out['DPS']*float(shares_outstanding)).round(0)
        out['Dividend_Yield_%'] = (out['DPS']/out['last_price']*100).round(2)
    if 'EPS' in out.columns:
        out['PER'] = (out['last_price']/out['EPS'].replace(0,np.nan)).replace([np.inf,-np.inf], np.nan).round(2)
    return out

def plot_fundamentals(ann_df: pd.DataFrame) -> go.Figure:
    x = ann_df['Annee']
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['Capitalisation (fin d’année)', 'Rendement annuel (%)',
                                        'Volatilité annualisée (%)', 'Volume annuel (titres)'],
                        vertical_spacing=0.16, horizontal_spacing=0.08)
    fig.add_trace(go.Bar(x=x, y=ann_df['market_cap_fin_annee_FCFA'], name='Capi fin année'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=ann_df['annual_return_%'], name='Rendement', mode='lines+markers'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=ann_df['vol_annual_%'], name='Vol annualisée', mode='lines+markers'), row=2, col=1)
    fig.add_trace(go.Bar(x=x, y=ann_df['vol_sum'], name='Volume annuel'), row=2, col=2)
    fig.update_yaxes(title_text="FCFA", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=2)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="Titres", row=2, col=2)
    fig.update_layout(height=480, showlegend=False, margin=dict(t=28,b=22,l=24,r=10))
    set_fig_template(fig); return fig

def plot_dividend_pe(ann_df: pd.DataFrame) -> Optional[go.Figure]:
    has_y = 'Dividend_Yield_%' in ann_df.columns and ann_df['Dividend_Yield_%'].notna().any()
    has_p = 'PER' in ann_df.columns and ann_df['PER'].notna().any()
    if not (has_y or has_p): return None
    x = ann_df['Annee']
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Dividend Yield (%)', 'PER (x)'],
                        vertical_spacing=0.06, horizontal_spacing=0.06)
    if has_y:
        fig.add_trace(go.Scatter(x=x, y=ann_df['Dividend_Yield_%'], mode='lines+markers', name='Yield'), row=1, col=1)
        fig.update_yaxes(title_text="%", row=1, col=1)
    if has_p:
        fig.add_trace(go.Scatter(x=x, y=ann_df['PER'], mode='lines+markers', name='PER'), row=1, col=2)
        fig.update_yaxes(title_text="x", row=1, col=2)
    fig.update_xaxes(title_text="Année", row=1, col=1)
    fig.update_xaxes(title_text="Année", row=1, col=2)
    fig.update_layout(height=360, showlegend=False, margin=dict(t=26,b=18,l=24,r=10))
    set_fig_template(fig); return fig

def summarize_fundamentals(ann_df: pd.DataFrame) -> str:
    if ann_df is None or ann_df.empty: return "Aucun indicateur fondamental calculable."
    df = ann_df.sort_values('Annee')
    first, last = df.iloc[0], df.iloc[-1]
    first_year, last_year = int(first['Annee']), int(last['Annee'])
    first_price, last_price = float(first['last_price']), float(last['last_price'])
    n_years = max(1, last_year - first_year)
    cagr = (last_price/first_price)**(1/n_years)-1 if first_price>0 else None
    lines = [f"**Synthèse fondamentale ({first_year}–{last_year})**",
             f"- **Prix fin {last_year}** : {last_price:,.2f} FCFA",
             f"- **Capitalisation fin {last_year}** : {int(last['market_cap_fin_annee_FCFA']):,} FCFA" if pd.notna(last['market_cap_fin_annee_FCFA']) else "",
             f"- **Rendement annuel {last_year}** : {float(last['annual_return_%']):.2f} %",
             f"- **Volatilité annualisée {last_year}** : {float(last['vol_annual_%']):.2f} %",
             f"- **Max Drawdown intra-année {last_year}** : {float(last['max_drawdown_intra_%']):.2f} %",
             f"- **Volume annuel moyen** : {df['vol_sum'].dropna().mean():,.0f} titres" if df['vol_sum'].notna().any() else "",
             f"- **CAGR ({first_year}→{last_year})** : {100*cagr:.2f} % / an" if cagr is not None else ""]
    if 'Dividend_Yield_%' in df.columns and pd.notna(df['Dividend_Yield_%'].iloc[-1]):
        lines.append(f"- **Rendement du dividende {last_year}** : {float(df['Dividend_Yield_%'].iloc[-1]):.2f} %")
    if 'Dividends_Total_FCFA' in df.columns and pd.notna(df['Dividends_Total_FCFA'].iloc[-1]):
        lines.append(f"- **Dividendes totaux {last_year}** : {float(df['Dividends_Total_FCFA'].iloc[-1]):,.0f} FCFA")
    if 'PER' in df.columns and pd.notna(df['PER'].iloc[-1]):
        lines.append(f"- **PER {last_year}** : {float(df['PER'].iloc[-1]):.2f}x")
    lines.append("> Capi = prix fin d’année × actions. EPS fourni ou estimé via DPS & payout ratio.")
    return "\n".join([l for l in lines if l])

# ====================== BACKTESTS (rapide) ======================
def backtest_sma(df, fast=20, slow=50, fee_bps=10.0, cash0=1_000_000.0):
    data = df[['Date','Close']].copy()
    data['SMA_fast'] = sma(data['Close'], fast)
    data['SMA_slow'] = sma(data['Close'], slow)
    data['signal'] = (data['SMA_fast'] > data['SMA_slow']).astype(int)
    data['trade'] = data['signal'].diff().fillna(0).astype(int)
    fee = fee_bps/10_000.0
    pos, cash, shares = 0, cash0, 0.0
    eq, trades = [], []
    for _, r in data.iterrows():
        px = r['Close']
        if r['trade']==1 and pos==0:
            shares = (cash*(1-fee))/px; cash=0.0; pos=1; trades.append((r['Date'], 'BUY', px, shares))
        elif r['trade']==-1 and pos==1:
            cash = shares*px*(1-fee); shares=0.0; pos=0; trades.append((r['Date'], 'SELL', px, 0.0))
        eq.append(cash + shares*px)
    data['equity'] = eq
    data['ret'] = data['equity'].pct_change().fillna(0)
    if len(data)>1:
        ann = _ann_factor('D')
        m, s = data['ret'].mean(), data['ret'].std()
        ann_ret = ((1+m)**ann - 1)*100 if m!=0 else 0.0
        ann_vol = s*np.sqrt(ann)*100 if s!=0 else 0.0
        sharpe = 0.0 if s==0 else (m/s)*np.sqrt(ann)
        mdd = (data['equity']/data['equity'].cummax() - 1).min()*100
    else:
        ann_ret = ann_vol = sharpe = mdd = 0.0
    stats = dict(
        capital_initial=cash0,
        capital_final=float(data['equity'].iloc[-1]) if len(data) else cash0,
        perf_totale_%=(float(data['equity'].iloc[-1])/cash0 - 1)*100 if len(data) else 0.0,
        perf_annualisee_%=ann_ret, vol_annualisee_%=ann_vol, sharpe=sharpe, max_drawdown_%=mdd
    )
    return data, stats, pd.DataFrame(trades, columns=['Date','Action','Prix','Quantite'])

# ====================== PREVISION — Sélection rapide meilleur modèle ======================
def _fit_es(train):
    try:
        model = ExponentialSmoothing(train, trend='add', seasonal=None, initialization_method="estimated")
        res = model.fit(optimized=True)
        return res, res.aic
    except Exception:
        return None, np.inf

def _fit_sarimax(train, order=(1,1,1)):
    try:
        res = SARIMAX(train, order=order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return res, res.aic
    except Exception:
        return None, np.inf

def choose_best_model(series: pd.Series):
    series = series.asfreq('D')  # daily régulier
    series = series.fillna(method='ffill')
    candidates = []

    es_res, es_aic = _fit_es(series)
    if es_res is not None:
        candidates.append(("Holt-Winters (additif)", es_res, es_aic))

    for o in [(1,1,0),(2,1,1),(1,1,1),(2,1,2)]:
        sr_res, sr_aic = _fit_sarimax(series, order=o)
        if sr_res is not None:
            candidates.append((f"SARIMAX{str(o)}", sr_res, sr_aic))

    if not candidates:
        return None, "Aucun modèle ajusté", None

    best = sorted(candidates, key=lambda x: x[2])[0]
    name, model, aic = best
    return model, name, aic

def forecast_best(df_daily: pd.DataFrame, horizon_days=90):
    s = df_daily.set_index('Date')['Close']
    s = s.asfreq('D').ffill()
    model, name, aic = choose_best_model(s)
    if model is None:
        return None, "Ajustement impossible", None
    pred = model.get_forecast(steps=horizon_days)
    fc = pred.predicted_mean
    conf = pred.conf_int(alpha=0.2) if hasattr(pred, 'conf_int') else None
    return (fc, conf), name, aic

def summarize_forecast(fc_series: pd.Series, conf_df: Optional[pd.DataFrame]) -> str:
    last_fc = float(fc_series.iloc[-1])
    first_fc = float(fc_series.iloc[0])
    change = (last_fc/first_fc - 1)*100 if first_fc>0 else 0.0
    msg = [f"**Résumé prédictif (horizon {len(fc_series)} jours)**",
           f"- Niveau attendu à la fin de l'horizon : **{last_fc:,.0f} FCFA**",
           f"- Variation projetée sur l'horizon : **{change:.1f} %**"]
    if conf_df is not None and {'lower Close','upper Close'}.issubset(set(conf_df.columns)):
        lo, hi = conf_df.iloc[-1]['lower Close'], conf_df.iloc[-1]['upper Close']
        msg.append(f"- Intervalle de confiance (80%) fin d'horizon : **[{lo:,.0f} ; {hi:,.0f}] FCFA**")
    msg.append("> Interprétez la tendance avec prudence : la liquidité BRVM et les événements idiosyncratiques peuvent créer des ruptures de régime.")
    return "\n".join(msg)

# ====================== APP ======================
def main():
    apply_dark_theme()

    # -------- Titre propre, centré via colonnes (pas de HTML custom) --------
    c1, c2, c3 = st.columns([1,3,1])
    with c2:
        st.title("Dashboard Marchés Boursiers – BRVM")

    # -------- SIDEBAR (contrôles globaux) --------
    with st.sidebar:
        st.header("Données prix")
        uploader = st.file_uploader("CSV de PRIX", type=['csv'], key="price_csv")
        if uploader is not None:
            df_original = load_data(uploader); st.success("Prix chargés.")
        else:
            if os.path.exists(DEFAULT_PRICE_PATH):
                df_original = load_data(DEFAULT_PRICE_PATH); st.info(f"Fichier par défaut : {DEFAULT_PRICE_PATH}")
            else:
                st.error("Importez un CSV de prix."); st.stop()

        shares = st.number_input("Actions en circulation", min_value=1, value=DEFAULT_SHARES_OUTSTANDING, step=1000)

        st.header("Période & Fréquence (globales)")
        freq = st.selectbox("Fréquence", ['Jour','Semaine','Mois'], index=0)
        freq_code = {'Jour':'D','Semaine':'W','Mois':'M'}[freq]
        dmin, dmax = df_original['Date'].min().date(), df_original['Date'].max().date()
        dr = st.date_input("Fenêtre d'analyse", value=(dmin, dmax), min_value=dmin, max_value=dmax)

        # stocker globalement pour tous les onglets
        if isinstance(dr, tuple):
            st.session_state['date_start'] = pd.to_datetime(dr[0])
            st.session_state['date_end']   = pd.to_datetime(dr[1])
        else:
            st.session_state['date_start'] = pd.to_datetime(dmin)
            st.session_state['date_end']   = pd.to_datetime(dmax)
        st.session_state['freq_code'] = freq_code

        st.header("Indicateurs techniques")
        indicators = st.multiselect("Sélection", ['MM','EMA','Bollinger','RSI','MACD'], default=['MM','RSI'])
        with st.expander("Paramètres", expanded=False):
            cpa, cpb = st.columns(2)
            with cpa:
                sma1 = st.slider("MM1", 5, 60, 20, 1)
                sma2 = st.slider("MM2", 10, 200, 50, 1)
                ema1 = st.slider("EMA", 5, 60, 20, 1)
                bb_window = st.slider("BB Fenêtre", 10, 60, 20, 1)
            with cpb:
                bb_std = st.slider("BB Écart", 1.0, 3.0, 2.0, 0.1)
                rsi_window = st.slider("RSI", 5, 30, 14, 1)
                macd_fast = st.slider("MACD Rapide", 5, 20, 12, 1)
                macd_slow = st.slider("MACD Lent", 20, 40, 26, 1)
        st.session_state['tech_params'] = dict(
            show_sma='MM' in indicators, sma1=sma1, sma2=sma2,
            show_ema='EMA' in indicators, ema1=ema1,
            show_bb='Bollinger' in indicators, bb_window=bb_window, bb_std=bb_std,
            show_rsi='RSI' in indicators, rsi_window=rsi_window,
            show_macd='MACD' in indicators, macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=9
        )

        st.header("Style & Risque")
        chart_type = st.radio("Type de graphique", ['Ligne','Chandelles'], index=1)
        rf = st.number_input("Taux sans risque (%)", value=2.0, step=0.5)
        st.session_state['chart_type'] = chart_type
        st.session_state['rf'] = rf

        st.header("Dividendes & Bénéfices (facultatif)")
        dps_uploader = st.file_uploader("CSV DPS par année", type=['csv'], key="dps_csv")
        eps_uploader = st.file_uploader("CSV EPS (ou Résultat net)", type=['csv'], key="eps_csv")
        st.subheader("Saisie manuelle (si pas de fichiers)")
        manual_dps   = st.number_input("DPS (dernière année)", min_value=0.0, value=0.0, step=1.0)
        manual_payout= st.number_input("Payout ratio (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        st.session_state['manual_dps'] = manual_dps if manual_dps>0 else None
        st.session_state['manual_payout'] = manual_payout if manual_payout>0 else None

    # -------- Appliquer filtre global --------
    sdate = st.session_state['date_start']; edate = st.session_state['date_end']
    freq_code = st.session_state['freq_code']
    params = st.session_state['tech_params']
    chart_type, rf = st.session_state['chart_type'], st.session_state['rf']

    df_filtered = df_original[(df_original['Date'] >= sdate) & (df_original['Date'] <= edate)].copy()
    df_view = resample_ohlcv(df_filtered, freq_code)
    df_view = add_indics(df_view, params)
    metrics = perf_metrics(df_view, rf_annual_pct=rf, freq_code=freq_code)
    ann_df = compute_fundamentals_from_daily(df_filtered, DEFAULT_SHARES_OUTSTANDING if DEFAULT_SHARES_OUTSTANDING else 1)

    # DPS/EPS (uploads > defaults)
    if dps_uploader is not None:
        dps_df = _parse_year_value_df(dps_uploader, ['DPS','dps','dividend_per_share','dividende'])
    elif os.path.exists(DEFAULT_DPS_PATH):
        dps_df = _parse_year_value_df(DEFAULT_DPS_PATH, ['DPS','dps','dividend_per_share','dividende'])
    else:
        dps_df = None

    if eps_uploader is not None:
        eps_or_net_df = _parse_year_value_df(eps_uploader, ['EPS','eps','net_income','resultat_net','rn','benefice','profit'])
    elif os.path.exists(DEFAULT_EPS_PATH):
        eps_or_net_df = _parse_year_value_df(DEFAULT_EPS_PATH, ['EPS','eps','net_income','resultat_net','rn','benefice','profit'])
    elif os.path.exists(DEFAULT_NET_PATH):
        eps_or_net_df = _parse_year_value_df(DEFAULT_NET_PATH, ['EPS','eps','net_income','resultat_net','rn','benefice','profit'])
    else:
        eps_or_net_df = None

    ann_df = enrich_with_dividends_eps(ann_df, DEFAULT_SHARES_OUTSTANDING, dps_df, eps_or_net_df,
                                       st.session_state['manual_dps'], st.session_state['manual_payout'])

    # -------- Onglets --------
    tab_dash, tab_pred = st.tabs(["📊 Tableau de bord", "🔮 Prédiction"])

    with tab_dash:
        st.subheader("Métriques principales")
        badge = {"D":"Jour","W":"Semaine","M":"Mois"}[freq_code]
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric(f"Prix ({badge})", f"{metrics['current_price']:.0f} FCFA")
        m2.metric("Rendement total", f"{metrics['total_return']:.1f}%")
        m3.metric("Rend. annualisé", f"{metrics['annualized_return']:.1f}%")
        m4.metric("Volatilité", f"{metrics['volatility']:.1f}%")
        m5.metric("Max DD", f"{metrics['max_drawdown']:.1f}%")
        m6.metric("Sharpe", f"{metrics['sharpe']:.2f}")
        st.caption(f"Période affichée : {df_view['Date'].min().date()} → {df_view['Date'].max().date()} | Dernière MAJ: {metrics['last_update']}")

        st.subheader("Graphique technique")
        st.plotly_chart(plot_tech(df_view, chart_type, params), use_container_width=True, config={"displaylogo": False})

        extra = plot_dividend_pe(ann_df)
        if extra is not None:
            st.subheader("Dividend Yield & PER")
            st.plotly_chart(extra, use_container_width=True, config={"displaylogo": False})

        st.subheader("Fondamentaux (annuels)")
        if not ann_df.empty:
            st.plotly_chart(plot_fundamentals(ann_df), use_container_width=True, config={"displaylogo": False})
            st.markdown(summarize_fundamentals(ann_df))
            st.download_button(
                "Télécharger fondamentaux (CSV)",
                ann_df.to_csv(index=False).encode('utf-8'),
                file_name="CFAOCI_fondamentaux_filtre.csv", mime="text/csv"
            )
        else:
            st.info("Aucun fondamental calculable sur la fenêtre.")

        st.subheader("Backtesting — Crossover MM 20/50 (exemple rapide)")
        bt_df, bt_stats, _ = backtest_sma(df_view, fast=20, slow=50, fee_bps=10.0)
        d1,d2,d3,d4,d5,d6 = st.columns(6)
        d1.metric("Capital initial", f"{1_000_000:,.0f} FCFA")
        d2.metric("Capital final", f"{bt_stats['capital_final']:,.0f} FCFA")
        d3.metric("Perf. totale", f"{bt_stats['perf_totale_%']:.1f}%")
        d4.metric("Perf. annualisée", f"{bt_stats['perf_annualisee_%']:.1f}%")
        d5.metric("Max DD", f"{bt_stats['max_drawdown_%']:.1f}%")
        d6.metric("Sharpe", f"{bt_stats['sharpe']:.2f}")
        eq = go.Figure(data=[go.Scatter(x=bt_df['Date'], y=bt_df['equity'], mode='lines', name='Équity')])
        eq.update_layout(height=280, margin=dict(t=6,b=6,l=6,r=6)); set_fig_template(eq)
        st.plotly_chart(eq, use_container_width=True, config={"displaylogo": False})

    with tab_pred:
        st.subheader("Sélection automatique du meilleur modèle (ES / SARIMAX)")
        if df_filtered.shape[0] < 60:
            st.warning("Fenêtre trop courte pour une prédiction robuste (≥ 60 jours recommandé).")
        result, name, aic = forecast_best(df_filtered, horizon_days=90)
        if result is None:
            st.error(name)
        else:
            (fc, conf) = result
            fc_df = fc.to_frame(name='Forecast'); fc_df.index.name='Date'
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Close'], mode='lines', name='Historique'))
            fig.add_trace(go.Scatter(x=fc.index, y=fc.values, mode='lines', name='Prévision'))
            if conf is not None and {'lower Close','upper Close'}.issubset(set(conf.columns)):
                fig.add_traces([
                    go.Scatter(x=conf.index, y=conf['upper Close'], line=dict(width=0), showlegend=False),
                    go.Scatter(x=conf.index, y=conf['lower Close'], fill='tonexty', name='IC 80%', opacity=0.15)
                ])
            fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=420)
            set_fig_template(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

            st.markdown(f"**Modèle retenu** : {name} (AIC = {aic:.1f})")
            st.markdown(summarize_forecast(fc, conf))

            st.download_button(
                "Télécharger la série de prévision (CSV)",
                fc_df.to_csv().encode('utf-8'),
                file_name="prevision_90j.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
