# memoire_gueri_dashboard.py — Thème sombre + Filtres globaux (période+fréquence) appliqués partout
# Fondamentaux + Backtests + Prévision (ARIMA(1,0,1)+GARCH(1,1) MENSUEL FIXE) + Simulation (MENSUEL FIXE) + Guide
# ------------------------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import os
import warnings
from typing import Union, Dict, Tuple, List, Optional
from math import sqrt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statsmodels
from statsmodels.tsa.arima.model import ARIMA

# GARCH (arch)
try:
    from arch.univariate import GARCH as ARCH_GARCH, ConstantMean
    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False

warnings.filterwarnings('ignore')

# --------------------------- CONFIG ---------------------------
st.set_page_config(
    page_title="Dashboard Marchés Boursiers - BRVM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_SHARES_OUTSTANDING = 181_371_900  # modifiable dans la sidebar

# Fichiers par défaut
DEFAULT_PRICE_PATH = "CFAOCI_filtre.csv"
DEFAULT_DPS_PATH   = "dps_exemple.csv"
DEFAULT_EPS_PATH   = "eps_exemple.csv"
DEFAULT_NET_PATH   = "net_income_exemple.csv"

# ===== CSS sombre compact + Titre lisible =====
BASE_CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 0.8rem; max-width: 1600px;}
section[data-testid="stSidebar"] .block-container {padding-top: 0.5rem; padding-bottom: 0.5rem;}
div[data-testid="stVerticalBlock"] {gap: 0.6rem;}
.element-container:has(.stPlotlyChart) {margin-bottom: 0.5rem;}
[data-testid="stMetric"] div {font-size: 0.92rem;}
[data-testid="stMetricValue"] {font-size: 1.22rem !important;}
[data-testid="stMetricDelta"] {font-size: 0.82rem !important;}
p, li { line-height: 1.38; font-size: 0.96rem; }
h2, h3, h4 { margin-bottom: 0.25rem; }
hr { margin: 0.6rem 0 0.7rem 0; }
.small-note { font-size: 0.9rem; color: #c9c7c4; }
.app-title { text-align: center; font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif; font-weight: 800; font-size: clamp(22px, 2.1vw, 30px); line-height: 1.12; margin: 0.7rem 0 0.45rem 0; -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; font-variant-ligatures: none; font-feature-settings: "liga" 0, "clig" 0, "kern" 1; text-rendering: optimizeLegibility; word-break: keep-all; overflow-wrap: anywhere; }
.app-subtitle { text-align: center; margin-top: -0.05rem; margin-bottom: 1.1rem; opacity: 0.92; font-size: clamp(12px, 1.08vw, 15px); }
body, .block-container { background-color: #0e1117; color: #e8e6e3; }
.app-subtitle { color: #c9c7c4; }
[data-testid="stMetric"] { background: #141823; border-radius: 10px; padding: 10px; border: 1px solid #1e2330; }
</style>
"""

def apply_dark_theme():
    st.markdown(BASE_CSS, unsafe_allow_html=True)

def centered_title(main: str, sub: str = ""):
    html = f'<h1 class="app-title">{main}</h1>'
    if sub:
        html += f'<div class="app-subtitle">{sub}</div>'
    st.markdown(html, unsafe_allow_html=True)

def set_fig_template(fig: go.Figure):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#e8e6e3", size=13),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#202533", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#202533", zeroline=False),
    )

# --------------------------- I/O & PARSING ---------------------------
@st.cache_data
def load_data(path_or_buffer: Union[str, io.BytesIO]) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'Dernier':'Close','Ouv.':'Open','Ouv':'Open',
        'Plus Haut':'High','Plus Bas':'Low','Vol.':'Volume','Variation %':'Variation'
    })
    if 'Date' not in df.columns:
        raise ValueError("Colonne 'Date' introuvable.")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])

    def to_num(x):
        if pd.isna(x): return np.nan
        s = str(x).replace('\u202f','').replace('\xa0','').replace(' ','').replace(',', '.')
        s = re.sub(r'[^0-9.\-]', '', s)
        return pd.to_numeric(s, errors='coerce')

    for col in ['Close','Open','High','Low']:
        if col in df.columns:
            df[col] = df[col].apply(to_num)

    def parse_volume(v):
        if pd.isna(v) or v == '': return 0.0
        s = str(v).strip().replace('\u202f','').replace('\xa0','').replace(' ','').replace(',', '.')
        m = re.match(r'^(-?\d+(\.\d+)?)([kKmM]?)$', s)
        if not m:
            s = re.sub(r'[^0-9.\-]','', s)
            return float(s) if s else 0.0
        val = float(m.group(1)); suf = m.group(3).lower()
        if suf == 'k': val *= 1_000
        if suf == 'm': val *= 1_000_000
        return val

    df['Volume'] = df['Volume'].apply(parse_volume) if 'Volume' in df.columns else 0.0
    if 'Variation' in df.columns:
        df['Variation'] = df['Variation'].apply(to_num)

    need = ['Date','Close','Open','High','Low']
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Colonne requise manquante: {c}")
    df = df.dropna(subset=need).sort_values('Date').reset_index(drop=True)
    # Cohérence OHLC
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

# --------------------------- INDICATEURS TECH ---------------------------
def calculate_sma(prices, window): return prices.rolling(window=window, min_periods=1).mean()
def calculate_ema(prices, window): return prices.ewm(span=window, adjust=False, min_periods=1).mean()

def calculate_rsi(prices: pd.Series, window: int = 14, method: str = "wilder") -> pd.Series:
    d = prices.diff(); up, dn = d.clip(lower=0), -d.clip(upper=0)
    if method == "wilder":
        ru = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        rd = dn.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    else:
        ru = up.rolling(window, min_periods=1).mean()
        rd = dn.rolling(window, min_periods=1).mean()
    rs = ru / rd.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.clip(0, 100).fillna(50)

def bollinger_bands(prices, window=20, n_std=2.0):
    ma = prices.rolling(window=window, min_periods=1).mean()
    sd = prices.rolling(window=window, min_periods=1).std(ddof=0)
    return ma - n_std*sd, ma, ma + n_std*sd

def macd(prices, fast=12, slow=26, signal=9):
    ef, es = calculate_ema(prices, fast), calculate_ema(prices, slow)
    line = ef - es
    sig = line.ewm(span=signal, adjust=False, min_periods=1).mean()
    hist = line - sig
    return line, sig, hist

def _annualization_factor(freq_code: str) -> float:
    return {'D': 252.0, 'W': 52.0, 'M': 12.0}.get(freq_code, 252.0)

def performance_metrics(df, rf_annual_pct=0.0, freq_code='D'):
    latest, oldest = df.iloc[-1], df.iloc[0]
    total_return = (latest['Close'] / oldest['Close'] - 1.0) * 100
    ann_fac = _annualization_factor(freq_code)
    ret = df['Close'].pct_change().dropna()
    if ret.empty:
        return {'current_price': latest['Close'], 'total_return': 0, 'annualized_return': 0,
                'volatility': 0, 'sharpe': 0, 'max_drawdown': 0, 'avg_volume': df['Volume'].mean(),
                'max_price': df['Close'].max(), 'min_price': df['Close'].min(),
                'last_update': latest['Date'].strftime('%d/%m/%Y')}
    mean_p, std_p = ret.mean(), ret.std()
    ann_return = ((1 + mean_p) ** ann_fac - 1) * 100
    vol = std_p * sqrt(ann_fac) * 100
    rf_per_period = (rf_annual_pct/100.0) / ann_fac
    sharpe = 0.0 if std_p == 0 else ((mean_p - rf_per_period) / std_p) * sqrt(ann_fac)
    cum = (1 + ret).cumprod()
    max_dd = (cum/cum.cummax() - 1).min() * 100
    return {
        'current_price': latest['Close'],
        'total_return': total_return,
        'annualized_return': ann_return,
        'volatility': vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'avg_volume': df['Volume'].mean(),
        'max_price': df['Close'].max(),
        'min_price': df['Close'].min(),
        'last_update': latest['Date'].strftime('%d/%m/%Y')
    }

def add_indicators(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    df = df.copy()
    if params.get('show_sma'):
        df['SMA_1'] = calculate_sma(df['Close'], params['sma1'])
        df['SMA_2'] = calculate_sma(df['Close'], params['sma2'])
    if params.get('show_ema'):
        df['EMA_1'] = calculate_ema(df['Close'], params['ema1'])
    if params.get('show_bb'):
        low, mid, up = bollinger_bands(df['Close'], params['bb_window'], params['bb_std'])
        df['BB_L'], df['BB_M'], df['BB_U'] = low, mid, up
    if params.get('show_rsi'):
        df['RSI'] = calculate_rsi(df['Close'], params['rsi_window'], method="wilder")
    if params.get('show_macd'):
        macd_l, macd_s, macd_h = macd(df['Close'], params['macd_fast'], params['macd_slow'], params['macd_signal'])
        df['MACD_L'], df['MACD_S'], df['MACD_H'] = macd_l, macd_s, macd_h
    return df

def plotly_combined_chart(df: pd.DataFrame, chart_type: str, params: Dict) -> go.Figure:
    rows = 1 + int(params.get('show_rsi')) + int(params.get('show_macd'))
    row_heights = [1.0] if rows==1 else ([0.68, 0.32] if rows==2 else [0.6, 0.22, 0.18])
    titles = ['Prix & Volume'] + (['RSI'] if params.get('show_rsi') else []) + (['MACD'] if params.get('show_macd') else [])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=row_heights, subplot_titles=titles)

    if chart_type == 'Chandelles':
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                     name='Cours', increasing_line_width=1.3, decreasing_line_width=1.3), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Prix', mode='lines', line=dict(width=2.4)), row=1, col=1)

    if params.get('show_sma'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_1'], name=f"MM{params['sma1']}", mode='lines', line=dict(width=1.6)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_2'], name=f"MM{params['sma2']}", mode='lines', line=dict(width=1.6)), row=1, col=1)
    if params.get('show_ema'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_1'], name=f"EMA{params['ema1']}", mode='lines', line=dict(width=1.4, dash='dot')), row=1, col=1)
    if params.get('show_bb'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_M'], name="BB moyenne", mode='lines', line=dict(dash='dot', width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_U'], showlegend=False, mode='lines', line=dict(width=0)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_L'], fill='tonexty', mode='lines', line=dict(width=0), name='BB Zone', opacity=0.08), row=1, col=1)

    current_row = 2
    if params.get('show_rsi'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', mode='lines', line=dict(width=1.6)), row=current_row, col=1)
        for y, dash, color in [(70,"dash","red"), (50,"dot","gray"), (30,"dash","green")]:
            fig.add_shape(type="line", xref=f"x{current_row}", yref=f"y{current_row}",
                          x0=df['Date'].min(), x1=df['Date'].max(), y0=y, y1=y,
                          line=dict(dash=dash, width=1, color=color))
        current_row += 1
    if params.get('show_macd'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_L'], name='MACD', mode='lines', line=dict(width=1.6)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_S'], name='Signal', mode='lines', line=dict(width=1.2, dash='dot')), row=current_row, col=1)
        fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_H'], name='Hist', opacity=0.55), row=current_row, col=1)

    fig.update_layout(height=640, showlegend=True,
                      legend=dict(orientation='h', yanchor='top', y=-0.14, xanchor='left', x=0),
                      margin=dict(t=30, b=60, l=28, r=22))
    set_fig_template(fig)
    return fig

# --------------------------- BACKTESTS ---------------------------
def backtest_sma(df, fast=20, slow=50, fee_bps=10.0, cash0=1_000_000.0):
    data = df[['Date','Close']].copy()
    data['SMA_fast'] = calculate_sma(data['Close'], fast)
    data['SMA_slow'] = calculate_sma(data['Close'], slow)
    data['signal'] = (data['SMA_fast'] > data['SMA_slow']).astype(int)
    data['signal_shift'] = data['signal'].shift(1).fillna(0).astype(int)
    data['trade'] = data['signal'] - data['signal_shift']
    fee = fee_bps / 10_000.0
    position, cash, shares = 0, cash0, 0.0
    equity_list, trades = [], []
    for _, r in data.iterrows():
        px = r['Close']
        if r['trade'] == 1 and position == 0:
            shares = (cash * (1 - fee)) / px; cash, position = 0.0, 1
            trades.append((r['Date'], 'BUY', px, shares))
        elif r['trade'] == -1 and position == 1:
            cash = shares * px * (1 - fee); shares, position = 0.0, 0
            trades.append((r['Date'], 'SELL', px, 0.0))
        equity_list.append(cash + shares * px)
    data['equity'] = equity_list
    data['ret'] = data['equity'].pct_change().fillna(0.0)
    if len(data) > 1:
        ann_fac = _annualization_factor('D')
        r_bar, s_bar = data['ret'].mean(), data['ret'].std()
        ann_ret = ((1 + r_bar) ** ann_fac - 1) * 100 if r_bar != 0 else 0.0
        ann_vol = s_bar * sqrt(ann_fac) * 100 if s_bar != 0 else 0.0
        sharpe = 0.0 if s_bar == 0 else (r_bar / s_bar) * sqrt(ann_fac)
        mdd = (data['equity']/data['equity'].cummax() - 1).min() * 100
    else:
        ann_ret = ann_vol = sharpe = mdd = 0.0
    stats = {
        'capital_initial': cash0,
        'capital_final': float(data['equity'].iloc[-1]) if len(data) else cash0,
        'perf_totale_%': (float(data['equity'].iloc[-1]) / cash0 - 1) * 100 if len(data) else 0.0,
        'perf_annualisee_%': ann_ret, 'vol_annualisee_%': ann_vol,
        'sharpe': sharpe, 'max_drawdown_%': mdd, 'nb_trades': int((data['trade'] != 0).sum() // 2)
    }
    trades_df = pd.DataFrame(trades, columns=['Date','Action','Prix','Quantite'])
    return data, stats, trades_df

def backtest_rsi_macd(df, rsi_window=14, rsi_buy=30.0, rsi_confirm=50.0, rsi_sell=70.0,
                      macd_fast=12, macd_slow=26, macd_signal=9, fee_bps=10.0, cash0=1_000_000.0):
    data = df[['Date','Close']].copy()
    data['RSI'] = calculate_rsi(data['Close'], rsi_window, method="wilder")
    m_l, m_s, _ = macd(data['Close'], macd_fast, macd_slow, macd_signal)
    data['MACD_L'], data['MACD_S'] = m_l, m_s
    prep_flag, prep_list = False, []
    for r in data.itertuples(index=False):
        if r.RSI < rsi_buy: prep_flag = True; prep_list.append(0)
        elif r.RSI >= rsi_confirm and prep_flag: prep_list.append(1); prep_flag = False
        else: prep_list.append(0)
    data['prep'] = prep_list
    data['macd_cross_up'] = ((data['MACD_L'].shift(1) <= data['MACD_S'].shift(1)) & (data['MACD_L'] > data['MACD_S'])).astype(int)
    data['macd_cross_down'] = ((data['MACD_L'].shift(1) >= data['MACD_S'].shift(1)) & (data['MACD_L'] < data['MACD_S'])).astype(int)
    data['buy_signal'] = ((data['prep'] == 1) & (data['macd_cross_up'] == 1)).astype(int)
    data['sell_signal'] = ((data['RSI'] > rsi_sell) | (data['macd_cross_down'] == 1)).astype(int)
    fee = fee_bps / 10_000.0
    position, cash, shares = 0, cash0, 0.0
    equity_list, trades = [], []
    for _, r in data.iterrows():
        px = r['Close']
        if position == 0 and r['buy_signal'] == 1:
            shares = (cash * (1 - fee)) / px; cash, position = 0.0, 1
            trades.append((r['Date'], 'BUY', px, shares))
        elif position == 1 and r['sell_signal'] == 1:
            cash = shares * px * (1 - fee); shares, position = 0.0, 0
            trades.append((r['Date'], 'SELL', px, 0.0))
        equity_list.append(cash + shares * px)
    data['equity'] = equity_list
    data['ret'] = data['equity'].pct_change().fillna(0.0)
    if len(data) > 1:
        ann_fac = _annualization_factor('D')
        r_bar, s_bar = data['ret'].mean(), data['ret'].std()
        ann_ret = ((1 + r_bar) ** ann_fac - 1) * 100 if r_bar != 0 else 0.0
        ann_vol = s_bar * sqrt(ann_fac) * 100 if s_bar != 0 else 0.0
        sharpe = 0.0 if s_bar == 0 else (r_bar / s_bar) * sqrt(ann_fac)
        mdd = (data['equity']/data['equity'].cummax() - 1).min() * 100
    else:
        ann_ret = ann_vol = sharpe = mdd = 0.0
    stats = {
        'capital_initial': cash0,
        'capital_final': float(data['equity'].iloc[-1]) if len(data) else cash0,
        'perf_totale_%': (float(data['equity'].iloc[-1]) / cash0 - 1) * 100 if len(data) else 0.0,
        'perf_annualisee_%': ann_ret, 'vol_annualisee_%': ann_vol,
        'sharpe': sharpe, 'max_drawdown_%': mdd, 'nb_trades': int((data['buy_signal'] == 1).sum())
    }
    trades_df = pd.DataFrame(trades, columns=['Date','Action','Prix','Quantite'])
    return data, stats, trades_df

def backtest_mixed_sma_rsi(df, sma_fast=20, sma_slow=50, rsi_window=14, rsi_enter=55.0, rsi_exit=45.0,
                           fee_bps=10.0, cash0=1_000_000.0):
    data = df[['Date','Close']].copy()
    data['SMA_fast'] = calculate_sma(data['Close'], sma_fast)
    data['SMA_slow'] = calculate_sma(data['Close'], sma_slow)
    data['RSI'] = calculate_rsi(data['Close'], rsi_window, method="wilder")
    data['enter'] = ((data['SMA_fast'] > data['SMA_slow']) & (data['RSI'] > rsi_enter)).astype(int)
    data['exit']  = ((data['SMA_fast'] < data['SMA_slow']) | (data['RSI'] < rsi_exit)).astype(int)
    fee = fee_bps / 10_000.0
    position, cash, shares = 0, cash0, 0.0
    equity_list, trades = [], []
    for _, r in data.iterrows():
        px = r['Close']
        if position == 0 and r['enter'] == 1:
            shares = (cash * (1 - fee)) / px; cash, position = 0.0, 1
            trades.append((r['Date'], 'BUY', px, shares))
        elif position == 1 and r['exit'] == 1:
            cash = shares * px * (1 - fee); shares, position = 0.0, 0
            trades.append((r['Date'], 'SELL', px, 0.0))
        equity_list.append(cash + shares * px)
    data['equity'] = equity_list
    data['ret'] = data['equity'].pct_change().fillna(0.0)
    if len(data) > 1:
        ann_fac = _annualization_factor('D')
        r_bar, s_bar = data['ret'].mean(), data['ret'].std()
        ann_ret = ((1 + r_bar) ** ann_fac - 1) * 100 if r_bar != 0 else 0.0
        ann_vol = s_bar * sqrt(ann_fac) * 100 if s_bar != 0 else 0.0
        sharpe = 0.0 if s_bar == 0 else (r_bar / s_bar) * sqrt(ann_fac)
        mdd = (data['equity']/data['equity'].cummax() - 1).min() * 100
    else:
        ann_ret = ann_vol = sharpe = mdd = 0.0
    stats = {
        'capital_initial': cash0,
        'capital_final': float(data['equity'].iloc[-1]) if len(data) else cash0,
        'perf_totale_%': (float(data['equity'].iloc[-1]) / cash0 - 1) * 100 if len(data) else 0.0,
        'perf_annualisee_%': ann_ret, 'vol_annualisee_%': ann_vol,
        'sharpe': sharpe, 'max_drawdown_%': mdd,
        'nb_trades': int(((data['enter'] == 1) & (data['exit'].shift(-1) == 1)).sum())
    }
    trades_df = pd.DataFrame(trades, columns=['Date','Action','Prix','Quantite'])
    return data, stats, trades_df

# --------------------------- FONDAMENTAUX ---------------------------
def _detect_year_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in ['Annee', 'Année', 'Year', 'year', 'period']:
        if c in df.columns:
            return c
    if isinstance(df.index, (pd.Int64Index, pd.UInt64Index, pd.RangeIndex)):
        df['Annee'] = df.index.astype(int)
        return 'Annee'
    for c in df.columns:
        s = df[c]
        try:
            vals = pd.to_numeric(s, errors='coerce')
            if vals.notna().mean() > 0.9 and (vals.between(1900, 2100)).mean() > 0.8:
                df.rename(columns={c: 'Annee'}, inplace=True)
                return 'Annee'
        except Exception:
            continue
    return None

def _year_span(df: pd.DataFrame) -> Optional[Tuple[int, int]]:
    if df is None or df.empty:
        return None
    col = _detect_year_column(df)
    if not col:
        return None
    try:
        y = pd.to_numeric(df[col], errors='coerce')
        return int(np.nanmin(y.values)), int(np.nanmax(y.values))
    except Exception:
        return None

@st.cache_data
def compute_market_fundamentals_from_original(df_original_filtered: pd.DataFrame, shares_outstanding: int) -> pd.DataFrame:
    if df_original_filtered.empty:
        return pd.DataFrame()
    df = df_original_filtered.copy().sort_values('Date').set_index('Date')
    df['ret'] = df['Close'].pct_change()
    ann = df.resample('Y').agg(
        last_price=('Close','last'),
        avg_price=('Close','mean'),
        vol_sum=('Volume','sum')
    )
    ann['last_close_prev'] = ann['last_price'].shift(1)
    ann['annual_return_%'] = ((ann['last_price']/ann['last_close_prev']) - 1.0) * 100

    def annual_vol(g):
        r = g['ret'].dropna()
        return (r.std() * np.sqrt(252) * 100) if len(r) > 1 else np.nan
    ann['vol_annual_%'] = df.groupby(pd.Grouper(freq='Y')).apply(annual_vol).values

    def max_dd(g):
        c = g['Close'].dropna()
        if c.empty: return np.nan
        return (c/c.cummax() - 1.0).min() * 100
    ann['max_drawdown_intra_%'] = df.groupby(pd.Grouper(freq='Y')).apply(max_dd).values

    ann['market_cap_fin_annee_FCFA'] = ann['last_price'] * float(shares_outstanding)

    years = ann.index.year.astype(int)
    ann = ann.reset_index(drop=True)
    ann.insert(0, 'Annee', years)

    ann['last_price'] = ann['last_price'].round(2)
    ann['avg_price'] = ann['avg_price'].round(2)
    ann['vol_sum'] = ann['vol_sum'].round(0).astype('Int64')
    ann['annual_return_%'] = ann['annual_return_%'].round(2)
    ann['vol_annual_%'] = ann['vol_annual_%'].round(2)
    ann['market_cap_fin_annee_FCFA'] = ann['market_cap_fin_annee_FCFA'].round(0).astype('Int64')
    ann['max_drawdown_intra_%'] = ann['max_drawdown_intra_%'].round(2)
    return ann

def _parse_year_value_df(uploaded_or_path, value_cols_candidates: List[str]) -> Optional[pd.DataFrame]:
    try:
        if isinstance(uploaded_or_path, (str, os.PathLike)):
            df = pd.read_csv(uploaded_or_path)
        else:
            df = pd.read_csv(uploaded_or_path)
    except Exception:
        return None
    df.columns = df.columns.str.strip()
    year_col = _detect_year_column(df)
    if not year_col:
        if 'period' in df.columns:
            df.rename(columns={'period': 'Annee'}, inplace=True)
            year_col = 'Annee'
        else:
            return None
    candidate = None
    for c in value_cols_candidates:
        if c in df.columns:
            candidate = c; break
    if candidate is None:
        lower = {c.lower(): c for c in df.columns}
        for c in value_cols_candidates:
            if c.lower() in lower:
                candidate = lower[c.lower()]; break
    if candidate is None:
        return None
    out = df[[year_col, candidate]].copy()
    out.rename(columns={year_col: 'Annee', candidate: candidate}, inplace=True)
    out['Annee'] = pd.to_numeric(out['Annee'], errors='coerce').astype('Int64')
    out[candidate] = pd.to_numeric(out[candidate], errors='coerce')
    out = out.dropna(subset=['Annee'])
    return out

def enrich_with_dividends_eps(ann_df: pd.DataFrame, shares_outstanding: int,
                              dps_df: Optional[pd.DataFrame],
                              eps_or_net_df: Optional[pd.DataFrame],
                              manual_dps: Optional[float], manual_payout_pct: Optional[float]) -> pd.DataFrame:
    if ann_df is None or ann_df.empty:
        return ann_df
    out = ann_df.copy()
    # DPS
    if dps_df is not None and not dps_df.empty:
        val_col = [c for c in dps_df.columns if c.lower() in ['dps','dividend_per_share','dividende','dividendes','dividende_par_action']]
        if val_col: dps_df = dps_df.rename(columns={val_col[0]: 'DPS'})
        elif 'DPS' not in dps_df.columns and dps_df.shape[1] == 2:
            other = [c for c in dps_df.columns if c != 'Annee'][0]; dps_df = dps_df.rename(columns={other:'DPS'})
        else: dps_df = None
        if dps_df is not None:
            out = out.merge(dps_df[['Annee','DPS']], on='Annee', how='left')
    # EPS / Net income
    if eps_or_net_df is not None and not eps_or_net_df.empty:
        eps_col = None; net_col = None
        for c in eps_or_net_df.columns:
            if c.lower() in ['eps','benefice_par_action','bnpa']: eps_col = c; break
        if eps_col is None:
            for c in eps_or_net_df.columns:
                if c.lower() in ['net_income','resultat_net','rn','benefice','profit']: net_col = c; break
        temp = eps_or_net_df.copy()
        if eps_col: temp = temp.rename(columns={eps_col:'EPS'})
        elif net_col: temp = temp.rename(columns={net_col:'net_income'})
        else: temp = None
        if temp is not None:
            out = out.merge(temp, on='Annee', how='left')
            if 'EPS' not in out.columns and 'net_income' in out.columns:
                out['EPS'] = out['net_income'] / float(shares_outstanding)

    if 'EPS' not in out.columns: out['EPS'] = np.nan

    if manual_dps is not None and manual_payout_pct is not None and len(out) > 0:
        try:
            last_year = int(out['Annee'].max())
            pay = max(min(manual_payout_pct/100.0, 0.9999), 0.0001)
            est_eps = manual_dps / pay
            out.loc[out['Annee'] == last_year, 'DPS'] = out.loc[out['Annee'] == last_year, 'DPS'].fillna(manual_dps)
            out.loc[out['Annee'] == last_year, 'EPS'] = out.loc[out['Annee'] == last_year, 'EPS'].fillna(est_eps)
        except Exception:
            pass

    if 'DPS' in out.columns:
        out['Dividends_Total_FCFA'] = (out['DPS'] * float(shares_outstanding)).round(0)
        out['Dividend_Yield_%'] = (out['DPS'] / out['last_price'] * 100).round(2)
    if 'EPS' in out.columns:
        out['PER'] = (out['last_price'] / out['EPS'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).round(2)
    return out

def plot_market_fundamentals_summary(ann_df: pd.DataFrame) -> go.Figure:
    year_col = _detect_year_column(ann_df) or 'Annee'
    x = ann_df[year_col]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Capitalisation (fin d’année)', 'Rendement annuel (%)',
                        'Volatilité annualisée (%)', 'Volume annuel (titres)'],
        vertical_spacing=0.16, horizontal_spacing=0.08
    )
    fig.add_trace(go.Bar(x=x, y=ann_df['market_cap_fin_annee_FCFA'], name='Capi fin année'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=ann_df['annual_return_%'], name='Rendement annuel', mode='lines+markers'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=ann_df['vol_annual_%'], name='Vol annualisée', mode='lines+markers'), row=2, col=1)
    fig.add_trace(go.Bar(x=x, y=ann_df['vol_sum'], name='Volume annuel'), row=2, col=2)
    fig.update_yaxes(title_text="FCFA", row=1, col=1)
    fig.update_yaxes(title_text="%",    row=1, col=2)
    fig.update_yaxes(title_text="%",    row=2, col=1)
    fig.update_yaxes(title_text="Titres", row=2, col=2)
    fig.update_layout(height=480, showlegend=False, margin=dict(t=28, b=22, l=24, r=10))
    set_fig_template(fig)
    return fig

def plot_dividend_and_pe(ann_df: pd.DataFrame) -> Optional[go.Figure]:
    if ann_df is None or ann_df.empty:
        return None
    year_col = _detect_year_column(ann_df) or 'Annee'
    x = ann_df[year_col]
    has_yield = 'Dividend_Yield_%' in ann_df.columns and ann_df['Dividend_Yield_%'].notna().any()
    has_per   = 'PER' in ann_df.columns and ann_df['PER'].notna().any()
    if not has_yield and not has_per:
        return None
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Dividend Yield (%)', 'PER (x)'],
                        shared_xaxes=False, vertical_spacing=0.06, horizontal_spacing=0.06)
    if has_yield:
        fig.add_trace(go.Scatter(x=x, y=ann_df['Dividend_Yield_%'], mode='lines+markers', name='Dividend Yield (%)', line=dict(width=2)), row=1, col=1)
        fig.update_yaxes(title_text="%", row=1, col=1)
    if has_per:
        fig.add_trace(go.Scatter(x=x, y=ann_df['PER'], mode='lines+markers', name='PER (x)', line=dict(width=2)), row=1, col=2)
        fig.update_yaxes(title_text="x", row=1, col=2)
    fig.update_xaxes(title_text="Année", row=1, col=1)
    fig.update_xaxes(title_text="Année", row=1, col=2)
    fig.update_layout(height=380, showlegend=False, margin=dict(t=26, b=18, l=24, r=10))
    set_fig_template(fig)
    return fig

def summarize_fundamentals(ann_df: pd.DataFrame) -> str:
    if ann_df is None or ann_df.empty:
        return "Aucun indicateur fondamental calculable sur la période importée."
    yc = _detect_year_column(ann_df) or 'Annee'
    ann_df = ann_df.sort_values(yc).reset_index(drop=True)
    last = ann_df.iloc[-1]
    last_year = int(last[yc]); last_price = float(last['last_price'])
    last_cap = int(last['market_cap_fin_annee_FCFA']) if pd.notna(last['market_cap_fin_annee_FCFA']) else None
    last_ret = float(last['annual_return_%']) if pd.notna(last['annual_return_%']) else None
    last_vol = float(last['vol_annual_%']) if pd.notna(last['vol_annual_%']) else None
    last_mdd = float(last['max_drawdown_intra_%']) if pd.notna(last['max_drawdown_intra_%']) else None
    vol_mean = ann_df['vol_sum'].dropna(); vol_mean = float(vol_mean.mean()) if not vol_mean.empty else None
    first = ann_df.iloc[0]; first_year = int(first[yc]); first_price = float(first['last_price'])
    n_years = max(1, last_year - first_year)
    cagr = None
    if first_price > 0: cagr = (last_price / first_price) ** (1 / n_years) - 1
    div_yield = ann_df['Dividend_Yield_%'].iloc[-1] if 'Dividend_Yield_%' in ann_df.columns else None
    div_total = ann_df['Dividends_Total_FCFA'].iloc[-1] if 'Dividends_Total_FCFA' in ann_df.columns else None
    per_last  = ann_df['PER'].iloc[-1] if 'PER' in ann_df.columns else None
    lines = []
    lines.append(f"**Synthèse fondamentale ({first_year}–{last_year})**")
    lines.append(f"- **Prix fin {last_year}** : {last_price:,.2f} FCFA")
    if last_cap is not None: lines.append(f"- **Capitalisation fin {last_year}** : {last_cap:,.0f} FCFA")
    if last_ret is not None: lines.append(f"- **Rendement annuel {last_year}** : {last_ret:.2f} %")
    if last_vol is not None: lines.append(f"- **Volatilité annualisée {last_year}** : {last_vol:.2f} %")
    if last_mdd is not None: lines.append(f"- **Max Drawdown intra-année {last_year}** : {last_mdd:.2f} %")
    if vol_mean is not None: lines.append(f"- **Volume annuel moyen (titres)** : {vol_mean:,.0f}")
    if cagr is not None: lines.append(f"- **CAGR ({first_year}→{last_year})** : {100*cagr:.2f} % / an")
    if pd.notna(div_yield): lines.append(f"- **Rendement du dividende {last_year}** : {float(div_yield):.2f} %")
    if pd.notna(div_total): lines.append(f"- **Dividendes totaux {last_year}** : {float(div_total):,.0f} FCFA")
    if pd.notna(per_last):  lines.append(f"- **PER {last_year}** : {float(per_last):.2f}x")
    lines.append("> Capi = prix fin d’année × actions. EPS fourni/calculé ou estimé via DPS & payout ratio.")
    return "\n".join(lines)

def describe_market_regimes(ann_df: pd.DataFrame) -> List[str]:
    if ann_df is None or ann_df.empty: return ["Aucune donnée annuelle disponible pour décrire les régimes de marché."]
    yc = _detect_year_column(ann_df) or 'Annee'
    df = ann_df[[yc, 'annual_return_%']].dropna().copy()
    if df.empty: return ["Rendements annuels indisponibles."]
    start, end = int(df[yc].min()), int(df[yc].max())
    windows = [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2025)]
    out = []
    for a, b in windows:
        s, e = max(start, a), min(end, b)
        if s > e:
            continue
        block = df[(df[yc] >= s) & (df[yc] <= e)]
        if block.empty:
            continue
        mean_ret = block['annual_return_%'].mean()
        best_row = block.loc[block['annual_return_%'].idxmax()]
        worst_row = block.loc[block['annual_return_%'].idxmin()]
        if mean_ret > 8:
            label = "marché haussier"
        elif mean_ret < -5:
            label = "marché baissier"
        else:
            label = "phase de consolidation"
        out.append(
            f"**{s}–{e}** : {label} (rendement moyen ≈ {mean_ret:.1f}%). "
            f"Meilleure année : {int(best_row[yc])} ({best_row['annual_return_%']:.1f}%). "
            f"Pire année : {int(worst_row[yc])} ({worst_row['annual_return_%']:.1f}%)."
        )
    if not out:
        out = [f"Période couverte {start}–{end} sans bloc standard complet ; tendance moyenne ≈ {df['annual_return_%'].mean():.1f}%."]
    return out

# --------------------------- OUTILS & PRÉVISION (MENSUEL FIXE) ---------------------------
def format_pct_scientific(x: float) -> str:
    try:
        return f"{float(x):.2e}%"
    except Exception:
        return str(x)

def compute_log_returns(close: pd.Series) -> pd.Series:
    close = close.astype(float)
    return np.log(close).diff().dropna()

def fit_arima101_on_returns(r_log: pd.Series):
    model = ARIMA(r_log, order=(1,0,1))
    res = model.fit(method_kwargs={"warn_convergence": False})
    return res

def fit_garch11_on_residuals(residuals: pd.Series):
    if not ARCH_AVAILABLE or len(residuals) < 60:
        return None, None
    am = ConstantMean(residuals.dropna())
    am.volatility = ARCH_GARCH(1,1)
    garch_res = am.fit(disp="off")
    return am, garch_res

def forecast_price_path_from_arima_garch(last_price: float, r_arima_res, garch_model, garch_res, horizon: int, seed: Optional[int] = None):
    mu_fc = r_arima_res.get_forecast(steps=horizon).predicted_mean.values
    if (garch_model is not None) and (garch_res is not None):
        vf = garch_res.forecast(horizon=horizon, reindex=False).variance.values.flatten()
        sigma = np.sqrt(np.maximum(vf, 1e-12))
    else:
        sigma = np.repeat(np.std(r_arima_res.resid), horizon)
    lo_r = mu_fc - 1.28*sigma
    hi_r = mu_fc + 1.28*sigma
    price_fc = np.empty(horizon); price_lo = np.empty(horizon); price_hi = np.empty(horizon)
    p = last_price; plo = last_price; phi = last_price
    for t in range(horizon):
        p  *= np.exp(mu_fc[t])
        plo*= np.exp(lo_r[t])
        phi*= np.exp(hi_r[t])
        price_fc[t], price_lo[t], price_hi[t] = p, plo, phi
    return price_fc, price_lo, price_hi

def forecast_figure(history: pd.DataFrame, y_col: str, pred: np.ndarray, horizon: int, bands=None, title: str = "Prévision"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history['Date'], y=history[y_col], mode='lines', name='Historique', line=dict(width=2.6)))
    last_date = pd.to_datetime(history['Date'].iloc[-1])
    future_idx = pd.date_range(last_date, periods=horizon+1, freq='M')[1:]
    if bands is not None:
        lo, hi = bands
        if len(lo) == horizon and len(hi) == horizon:
            fig.add_trace(go.Scatter(x=future_idx, y=hi, line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=future_idx, y=lo, fill='tonexty', name='Intervalle', opacity=0.18, line=dict(width=0)))
    fig.add_trace(go.Scatter(x=future_idx, y=pred, mode='lines+markers', name='Prévision', line=dict(width=2.8)))
    x_min = pd.to_datetime(history['Date'].min())
    x_max = pd.to_datetime(future_idx[-1]) if len(future_idx) else pd.to_datetime(history['Date'].max())
    fig.update_xaxes(range=[x_min, x_max], dtick="M12", tickformat="%Y", ticklabelmode="period")
    fig.update_layout(title=title, height=460,
                      margin=dict(t=52,b=80,l=24,r=12),
                      legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='left', x=0))
    set_fig_template(fig)
    return fig

# --------------------------- SIMULATION (MENSUEL FIXE) ---------------------------
def monte_carlo_arima_garch(last_price: float, r_arima_res, garch_model, garch_res,
                            horizon: int = 12, n_paths: int = 5000, scenario: str = "Neutre", seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    mu = r_arima_res.get_forecast(steps=horizon).predicted_mean.values.astype(float)
    if (garch_model is not None) and (garch_res is not None):
        vf = garch_res.forecast(horizon=horizon, reindex=False).variance.values.flatten()
        sigma = np.sqrt(np.maximum(vf, 1e-12)).astype(float)
    else:
        sigma = np.repeat(np.std(r_arima_res.resid), horizon).astype(float)
    if scenario.lower().startswith("opt"):
        mu_adj = mu * 1.25; sigma_adj = sigma * 0.9
    elif scenario.lower().startswith("pes"):
        mu_adj = mu * 0.60; sigma_adj = sigma * 1.25
    else:
        mu_adj = mu.copy(); sigma_adj = sigma.copy()
    eps = rng.standard_normal(size=(n_paths, horizon))
    r_sim = mu_adj + sigma_adj * eps
    prices = np.empty((n_paths, horizon))
    prices[:, 0] = last_price * np.exp(r_sim[:, 0])
    for t in range(1, horizon):
        prices[:, t] = prices[:, t-1] * np.exp(r_sim[:, t])
    return prices, r_sim

def var_cvar(returns: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    losses = -returns
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    cvar = tail.mean() if tail.size > 0 else var
    return var, cvar

def fan_chart_figure(base_dates: pd.DatetimeIndex, sims: np.ndarray, title: str) -> go.Figure:
    q = np.quantile(sims, [0.1, 0.5, 0.9], axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base_dates, y=q[2], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=base_dates, y=q[0], fill='tonexty', name='P10–P90', opacity=0.18, line=dict(width=0)))
    fig.add_trace(go.Scatter(x=base_dates, y=q[1], mode='lines+markers', name='Médiane (P50)', line=dict(width=2.8)))
    fig.update_layout(title=title, height=420, margin=dict(t=42,b=60,l=24,r=12),
                      legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='left', x=0))
    set_fig_template(fig)
    return fig

# --------------------------- GUIDE (onglet) ---------------------------
def guide_tab():
    st.markdown("## Guide & Méthodologie")
    with st.expander("Indicateurs techniques", expanded=True):
        st.markdown("""
- **MM (SMA)** : moyenne arithmétique des *Close*. Croisement **rapide>lente** = biais haussier.  
- **EMA** : moyenne exponentielle (réagit plus vite).  
- **Bandes de Bollinger** (20, ±2σ) : zone de prix “normale” autour de la MM.  
- **RSI (14)** : >70 surachat ; <30 survente ; 50 neutre.  
- **MACD (12/26/9)** : croisement MACD↑Signal = reprise haussière ; MACD↓Signal = essoufflement.
        """)
    with st.expander("Métriques de performance", expanded=False):
        st.markdown("""
- **Rendement total** ; **Annualisé** (selon fréquence) ; **Volatilité** ; **Sharpe** ; **Max Drawdown**.  
- **CAGR** et **Synthèse** dans la section fondamentaux.
        """)
    with st.expander("Backtests", expanded=False):
        st.markdown("**SMA Crossover**, **RSI+MACD**, **Mixte (SMA+RSI)** — frais (bps) inclus.")
    with st.expander("Modèles du mémoire", expanded=True):
        st.markdown("""
- **Moyenne conditionnelle** : **ARIMA(1,0,1)** sur les rendements logarithmiques.  
- **Volatilité conditionnelle** : **GARCH(1,1)**.  
- **Prévision** : trajectoire moyenne + intervalle d'incertitude (mensuel fixe).  
- **Simulation Monte-Carlo (12 mois, 5000 trajectoires)** avec **3 scénarios** et **VaR/CVaR** (mensuel fixe).  
        """)

# --------------------------- APP ---------------------------
def main():
    apply_dark_theme()
    centered_title("Dashboard Marchés Boursiers – BRVM",
                   "Filtres globaux | Analyse technique & fondamentale | Backtests | Prévision ARIMA(1,0,1)+GARCH(1,1) (mensuel) | Simulation (mensuel)")

    # État global
    if 'global_date_start' not in st.session_state:
        st.session_state.global_date_start = None
    if 'global_date_end' not in st.session_state:
        st.session_state.global_date_end = None
    if 'global_freq_code' not in st.session_state:
        st.session_state.global_freq_code = 'D'

    # Onglets
    tab_main, tab_forecast, tab_simul, tab_guide = st.tabs(["Tableau de bord", "Prévision", "Simulation", "Guide & Méthode"])

    # ===================== TAB PRINCIPAL =====================
    with tab_main:
        with st.sidebar:
            st.header("Données prix")
            uploader = st.file_uploader("Importer le CSV de PRIX", type=['csv'], key="price_csv_main")
            if uploader is not None:
                df_original = load_data(uploader)
                st.success("Données de prix chargées")
            else:
                if os.path.exists(DEFAULT_PRICE_PATH):
                    df_original = load_data(DEFAULT_PRICE_PATH)
                    st.info(f"Données de prix par défaut : {DEFAULT_PRICE_PATH}")
                else:
                    st.error("Aucun fichier de prix. Importez un CSV")
                    st.stop()

            shares = st.number_input("Actions en circulation (exactes)", min_value=1, value=DEFAULT_SHARES_OUTSTANDING, step=1000, key="shares_input")

            st.header("Période & Fréquence (GLOBAL)")
            freq = st.selectbox("Fréquence", ['Jour', 'Semaine', 'Mois'], index=0, key="freq_select",
                                help="Pilote TOUT le dashboard (sauf agrégats annuels).")
            st.session_state.global_freq_code = {'Jour':'D','Semaine':'W','Mois':'M'}[freq]

            dmin, dmax = df_original['Date'].min().date(), df_original['Date'].max().date()
            default_range = (
                st.session_state.global_date_start.date() if st.session_state.global_date_start is not None else dmin,
                st.session_state.global_date_end.date()   if st.session_state.global_date_end   is not None else dmax
            )
            dr = st.date_input("Fenêtre d'analyse (globale)", value=default_range, min_value=dmin, max_value=dmax,
                               help="Utilisée dans *Tableau de bord*, *Prévision* et *Simulation*.", key="date_window")
            if isinstance(dr, tuple):
                st.session_state.global_date_start = pd.to_datetime(dr[0])
                st.session_state.global_date_end   = pd.to_datetime(dr[1])
            else:
                st.session_state.global_date_start = pd.to_datetime(dmin)
                st.session_state.global_date_end   = pd.to_datetime(dmax)

            st.header("Indicateurs techniques")
            indicators = st.multiselect("Sélection", ['MM', 'EMA', 'Bollinger', 'RSI', 'MACD'], default=['MM','RSI'], key="indics_select")
            with st.expander("Paramètres", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    sma1 = st.slider("MM1", 5, 60, 20, 1, key="sma1_slider")
                    sma2 = st.slider("MM2", 10, 200, 50, 1, key="sma2_slider")
                    ema1 = st.slider("EMA", 5, 60, 20, 1, key="ema1_slider")
                    bb_window = st.slider("BB Fenêtre", 10, 60, 20, 1, key="bbwin_slider")
                with c2:
                    bb_std = st.slider("BB Écart", 1.0, 3.0, 2.0, 0.1, key="bbsd_slider")
                    rsi_window = st.slider("RSI", 5, 30, 14, 1, key="rsiw_slider")
                    macd_fast = st.slider("MACD Rapide", 5, 20, 12, 1, key="macdf_slider")
                    macd_slow = st.slider("MACD Lent", 20, 40, 26, 1, key="macds_slider")

            params = {
                'show_sma':'MM' in indicators, 'sma1':sma1, 'sma2':sma2,
                'show_ema':'EMA' in indicators, 'ema1':ema1,
                'show_bb':'Bollinger' in indicators, 'bb_window':bb_window, 'bb_std':bb_std,
                'show_rsi':'RSI' in indicators, 'rsi_window':rsi_window,
                'show_macd':'MACD' in indicators, 'macd_fast':macd_fast, 'macd_slow':macd_slow, 'macd_signal':9
            }

            st.header("Style & Risque")
            chart_type = st.radio("Type de graphique", ['Ligne','Chandelles'], key="chart_type_radio")
            rf = st.number_input("Taux sans risque (%)", value=2.0, step=0.5, key="rf_input")

            st.header("Backtesting")
            strat = st.selectbox("Stratégie", ["SMA Crossover", "RSI + MACD", "Mixte (SMA + RSI)"], key="strat_select")
            if strat == "SMA Crossover":
                b1,b2,b3 = st.columns(3)
                with b1: bt_fast = st.number_input("MM rapide", 2, 200, 20, 1, key="bt_sma_fast")
                with b2: bt_slow = st.number_input("MM lente", 5, 400, 50, 1, key="bt_sma_slow")
                with b3: bt_fee  = st.number_input("Frais (bps)", 0.0, 200.0, 10.0, 1.0, key="bt_sma_fee")
            elif strat == "RSI + MACD":
                b1,b2,b3 = st.columns(3)
                with b1: bt_rsi_buy = st.slider("RSI sous-achat", 10, 40, 30, 1, key="bt_rsi_buy")
                with b2: bt_rsi_confirm = st.slider("RSI confirmation", 30, 60, 50, 1, key="bt_rsi_confirm")
                with b3: bt_rsi_sell = st.slider("RSI surachat", 50, 90, 70, 1, key="bt_rsi_sell")
                c4,c5,c6 = st.columns(3)
                with c4: bt_macd_fast = st.slider("MACD rapide", 5, 20, 12, 1, key="bt_macd_fast")
                with c5: bt_macd_slow = st.slider("MACD lent", 20, 40, 26, 1, key="bt_macd_slow")
                with c6: bt_macd_signal = st.slider("MACD signal", 5, 20, 9, 1, key="bt_macd_signal")
                bt_fee = st.number_input("Frais (bps)", 0.0, 200.0, 10.0, 1.0, key="bt_rsi_fee")
            else:
                b1,b2,b3 = st.columns(3)
                with b1: mix_sma_fast = st.number_input("MM rapide", 2, 200, 20, 1, key="mix_sma_fast")
                with b2: mix_sma_slow = st.number_input("MM lente", 5, 400, 50, 1, key="mix_sma_slow")
                with b3: mix_fee = st.number_input("Frais (bps)", 0.0, 200.0, 10.0, 1.0, key="mix_fee")
                c4,c5 = st.columns(2)
                with c4: mix_rsi_enter = st.slider("RSI entrée", 40, 70, 55, 1, key="mix_rsi_enter")
                with c5: mix_rsi_exit  = st.slider("RSI sortie", 20, 60, 45, 1, key="mix_rsi_exit")

            st.header("Dividendes & Bénéfices (facultatif)")
            dps_uploader = st.file_uploader("CSV DPS par année", type=['csv'], key="dps_csv")
            eps_uploader = st.file_uploader("CSV EPS (ou Résultat net)", type=['csv'], key="eps_csv")

            st.subheader("Saisie manuelle (si pas de fichiers)")
            manual_dps   = st.number_input("DPS (dernière année)", min_value=0.0, value=0.0, step=1.0, key="manual_dps")
            manual_payout= st.number_input("Payout ratio (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="manual_payout")

        # ===== TRAITEMENTS GLOBAUX =====
        start_date = st.session_state.global_date_start
        end_date   = st.session_state.global_date_end
        # Base filtrée (JOUR)
        df_filtered_daily = df_original[(df_original['Date'] >= start_date) & (df_original['Date'] <= end_date)].copy()
        # Resample global à la fréquence choisie (D/W/M)
        df_resampled = resample_ohlcv(df_filtered_daily, st.session_state.global_freq_code)

        # Indicateurs + métriques sur la vue resamplée
        df_view = add_indicators(df_resampled, params)
        metrics = performance_metrics(df_view, rf_annual_pct=rf, freq_code=st.session_state.global_freq_code)

        # Fondamentaux annuels sur base FILTRÉE (toujours annuel)
        ann_df = compute_market_fundamentals_from_original(df_filtered_daily, shares)

        if dps_uploader is not None:
            dps_df = _parse_year_value_df(dps_uploader, ['DPS','dps','dividend_per_share','dividende','dividendes','dividende_par_action'])
        elif os.path.exists(DEFAULT_DPS_PATH):
            dps_df = _parse_year_value_df(DEFAULT_DPS_PATH, ['DPS','dps','dividend_per_share','dividende','dividendes','dividende_par_action'])
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

        manual_dps_val = manual_dps if manual_dps > 0 else None
        manual_payout_val = manual_payout if manual_payout > 0 else None
        ann_df = enrich_with_dividends_eps(ann_df, shares, dps_df, eps_or_net_df, manual_dps_val, manual_payout_val)

        span = _year_span(ann_df)
        fund_title_suffix = f"({span[0]}–{span[1]})" if span else "(n/a)"

        # ===== MÉTRIQUES =====
        st.subheader("Métriques principales")
        badge = {"D":"Jour","W":"Semaine","M":"Mois"}[st.session_state.global_freq_code]
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric(f"Prix ({badge})", f"{metrics['current_price']:.0f} FCFA")
        m2.metric("Rendement total", f"{metrics['total_return']:.1f}%")
        m3.metric("Rend. annualisé", format_pct_scientific(metrics['annualized_return']))
        m4.metric("Volatilité", f"{metrics['volatility']:.1f}%")
        m5.metric("Max DD", f"{metrics['max_drawdown']:.1f}%")
        m6.metric("Sharpe", f"{metrics['sharpe']:.2f}")
        st.caption(f"Période affichée : {df_view['Date'].min().date()} → {df_view['Date'].max().date()} | Dernière MAJ: {metrics['last_update']}")

        # ===== 1) GRAPHIQUE TECHNIQUE =====
        st.subheader("Graphique technique")
        tech_fig = plotly_combined_chart(df_view, chart_type, params)
        st.plotly_chart(tech_fig, use_container_width=True, config={"displaylogo": False})

        # ===== 2) Dividend Yield & PER (AUTO) =====
        extra_fig = plot_dividend_and_pe(ann_df)
        if extra_fig is not None:
            st.subheader("Dividend Yield & PER")
            st.plotly_chart(extra_fig, use_container_width=True, config={"displaylogo": False})

        # ===== 3) Graphiques fondamentaux =====
        st.subheader(f"Fondamentaux de marché {fund_title_suffix}")
        if (ann_df is not None) and (not ann_df.empty):
            fund_fig = plot_market_fundamentals_summary(ann_df)
            st.plotly_chart(fund_fig, use_container_width=True, config={"displaylogo": False})
            st.markdown(summarize_fundamentals(ann_df))
            st.markdown("**Régimes de marché (par périodes standards)**")
            for line in describe_market_regimes(ann_df):
                st.write(f"- {line}")
            fname = f"CFAOCI_fondamentaux_{span[0]}_{span[1]}.csv" if span else "CFAOCI_fondamentaux.csv"
            st.download_button(
                f"Télécharger fondamentaux {fund_title_suffix} (CSV)",
                ann_df.to_csv(index=False).encode('utf-8'),
                file_name=fname, mime="text/csv", key="dl_fonda"
            )
        else:
            st.info("Aucun fondamental calculable (fichier vide ou colonnes manquantes).")

        # ===== 4) BACKTEST =====
        st.subheader(f"Backtesting — {strat}")
        if len(df_view) < 10:
            st.warning("Période trop courte pour backtester.")
        else:
            if strat == "SMA Crossover":
                bt_df, bt_stats, bt_trades = backtest_sma(df_view, fast=int(bt_fast), slow=int(bt_slow), fee_bps=float(bt_fee))
            elif strat == "RSI + MACD":
                bt_df, bt_stats, bt_trades = backtest_rsi_macd(
                    df_view, rsi_window=int(rsi_window),
                    rsi_buy=float(bt_rsi_buy), rsi_confirm=float(bt_rsi_confirm), rsi_sell=float(bt_rsi_sell),
                    macd_fast=int(bt_macd_fast), macd_slow=int(bt_macd_slow), macd_signal=int(bt_macd_signal),
                    fee_bps=float(bt_fee)
                )
            else:
                bt_df, bt_stats, bt_trades = backtest_mixed_sma_rsi(
                    df_view, sma_fast=int(mix_sma_fast), sma_slow=int(mix_sma_slow),
                    rsi_window=int(rsi_window), rsi_enter=float(mix_rsi_enter), rsi_exit=float(mix_rsi_exit),
                    fee_bps=float(mix_fee)
                )

            d1,d2,d3,d4,d5,d6 = st.columns(6)
            d1.metric("Capital initial", f"{bt_stats['capital_initial']:,.0f} FCFA")
            d2.metric("Capital final", f"{bt_stats['capital_final']:,.0f} FCFA")
            d3.metric("Perf. totale", f"{bt_stats['perf_totale_%']:.1f}%")
            d4.metric("Perf. annualisée", format_pct_scientific(bt_stats['perf_annualisee_%']))
            d5.metric("Max DD", f"{bt_stats['max_drawdown_%']:.1f}%")
            d6.metric("Sharpe", f"{bt_stats['sharpe']:.2f}")

            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(x=bt_df['Date'], y=bt_df['equity'], mode='lines', name='Équity', line=dict(width=2.4)))
            eq_fig.update_layout(height=280, margin=dict(t=6,b=6,l=6,r=6))
            set_fig_template(eq_fig)
            st.plotly_chart(eq_fig, use_container_width=True, config={"displaylogo": False})

    # ===================== TAB PRÉVISION (Mensuel Fixe) =====================
    with tab_forecast:
        st.markdown("### Paramètres de prévision — ARIMA(1,0,1) + GARCH(1,1) (horizon mensuel fixe)")
        # Recharger la base si nécessaire (mêmes filtres globaux)
        if 'df_original' not in locals():
            if os.path.exists(DEFAULT_PRICE_PATH):
                df_original = load_data(DEFAULT_PRICE_PATH)
            else:
                st.error("Chargez d'abord les données dans l'onglet principal.")
                st.stop()

        start_f = st.session_state.global_date_start
        end_f   = st.session_state.global_date_end
        df_filtered_daily = df_original[(df_original['Date'] >= start_f) & (df_original['Date'] <= end_f)].copy()
        # Prévision TOUJOURS sur données mensuelles (mémoire)
        df_m = resample_ohlcv(df_filtered_daily, 'M')
        if len(df_m) < 24:
            st.warning("Au moins 24 points mensuels sont requis.")
            st.stop()

        s_close = df_m.set_index('Date')['Close'].astype(float)
        r_log = compute_log_returns(s_close)

        with st.spinner("Estimation ARIMA(1,0,1) sur les rendements…"):
            arima_res = fit_arima101_on_returns(r_log)
        with st.spinner("Estimation GARCH(1,1) sur les résidus…"):
            garch_model, garch_res = fit_garch11_on_residuals(arima_res.resid)

        horizon_months = st.number_input("Horizon (mois)", min_value=3, max_value=60, value=12, step=1, key="forecast_horizon_months")
        last_price = float(s_close.iloc[-1])

        price_fc, price_lo, price_hi = forecast_price_path_from_arima_garch(
            last_price, arima_res, garch_model, garch_res, int(horizon_months), seed=42
        )

        hist_df = pd.DataFrame({'Date': s_close.index, 'Close': s_close.values})
        fc_fig = forecast_figure(hist_df, 'Close', price_fc, int(horizon_months), bands=(price_lo, price_hi),
                                 title="Prévision (ARIMA(1,0,1) + GARCH(1,1)) — horizon mensuel")
        st.plotly_chart(fc_fig, use_container_width=True, config={"displaylogo": False})

        chg_avg = 100.0*(np.mean(price_fc)/last_price - 1.0)
        st.markdown(f"- Dernier prix : **{last_price:,.2f}** FCFA  |  Variation moyenne prévue (horizon) : **{chg_avg:.2f}%**")
        st.caption("Intervalle d'incertitude ≈ IC 80% (construction ±1.28σ sur rendements log).")

        future_idx = pd.date_range(s_close.index[-1], periods=int(horizon_months)+1, freq='M')[1:]
        out = pd.DataFrame({'Date': future_idx, 'Forecast': price_fc, 'Low_approx': price_lo, 'High_approx': price_hi})
        st.download_button("Télécharger la prévision (CSV)", out.to_csv(index=False).encode('utf-8'),
                           file_name="prevision_arima_garch.csv", mime="text/csv", key="dl_forecast")

        if not ARCH_AVAILABLE:
            st.caption("Astuce : pour activer **GARCH**, installez le paquet `arch` (sinon écart-type fixe des résidus ARIMA).")

    # ===================== TAB SIMULATION (Mensuel Fixe) =====================
    with tab_simul:
        st.markdown("### Simulation Monte-Carlo — 12 mois, 5000 trajectoires, scénarios (mensuel fixe)")
        st.write("Méthode : ARIMA(1,0,1) pour la moyenne des rendements log, GARCH(1,1) pour la volatilité conditionnelle. Mesures de risque VaR/CVaR.")

        if 'df_original' not in locals():
            if os.path.exists(DEFAULT_PRICE_PATH):
                df_original = load_data(DEFAULT_PRICE_PATH)
            else:
                st.error("Chargez d'abord les données dans l'onglet principal.")
                st.stop()

        start_s = st.session_state.global_date_start
        end_s   = st.session_state.global_date_end
        df_filtered_daily = df_original[(df_original['Date'] >= start_s) & (df_original['Date'] <= end_s)].copy()
        df_m = resample_ohlcv(df_filtered_daily, 'M')
        if len(df_m) < 24:
            st.warning("Au moins 24 points mensuels sont requis.")
            st.stop()

        s_close = df_m.set_index('Date')['Close'].astype(float)
        r_log = compute_log_returns(s_close)

        with st.spinner("Estimation ARIMA(1,0,1) & GARCH(1,1)…"):
            arima_res = fit_arima101_on_returns(r_log)
            garch_model, garch_res = fit_garch11_on_residuals(arima_res.resid)

        colA, colB, colC = st.columns(3)
        with colA:
            horizon_mc = st.number_input("Horizon (mois)", 3, 60, 12, 1, key="mc_horizon")
        with colB:
            n_paths = st.number_input("Nb trajectoires", 100, 20000, 5000, 100, key="mc_paths")
        with colC:
            seed = st.number_input("Graine aléatoire", 0, 10_000, 42, 1, key="mc_seed")

        last_price = float(s_close.iloc[-1])
        base_idx = pd.date_range(s_close.index[-1], periods=int(horizon_mc)+1, freq='M')[1:]

        scenarios = ["Optimiste", "Neutre", "Pessimiste"]
        sim_results = {}
        for sc in scenarios:
            sims, r_sim = monte_carlo_arima_garch(
                last_price, arima_res, garch_model, garch_res,
                horizon=int(horizon_mc), n_paths=int(n_paths), scenario=sc, seed=int(seed)
            )
            sim_results[sc] = (sims, r_sim)

        st.subheader("Fan-charts (P10–P90) par scénario")
        for sc in scenarios:
            sims, _ = sim_results[sc]
            fig = fan_chart_figure(base_idx, sims, f"Scénario {sc}")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        st.subheader("Synthèse à l’horizon")
        rows = []
        for sc in scenarios:
            sims, _ = sim_results[sc]
            pT = sims[:, -1]
            ret = pT/last_price - 1.0
            p10, p50, p90 = np.quantile(pT, [0.10, 0.50, 0.90])
            var95, cvar95 = var_cvar(ret, alpha=0.95)
            rows.append({
                "Scénario": sc,
                "P10 (FCFA)": round(p10, 2),
                "P50 (FCFA)": round(p50, 2),
                "P90 (FCFA)": round(p90, 2),
                "VaR95 (perte %)": round(100*var95, 2),
                "CVaR95 (perte %)": round(100*cvar95, 2)
            })
        df_syn = pd.DataFrame(rows)
        st.dataframe(df_syn, use_container_width=True)

        out_rows = []
        for sc in scenarios:
            sims, _ = sim_results[sc]
            q = np.quantile(sims, [0.10,0.50,0.90], axis=0)
            df_q = pd.DataFrame({
                "Date": base_idx, f"{sc}_P10": q[0], f"{sc}_P50": q[1], f"{sc}_P90": q[2]
            })
            out_rows.append(df_q.set_index("Date"))
        out_all = pd.concat(out_rows, axis=1).reset_index()
        st.download_button("Télécharger quantiles simulés (CSV)", out_all.to_csv(index=False).encode('utf-8'),
                           file_name="simulations_quantiles.csv", mime="text/csv", key="dl_mc")

        st.caption("Remarque : VaR/CVaR calculées sur les rendements cumulés à l'horizon (pertes positives).")

    # ===================== TAB GUIDE =====================
    with tab_guide:
        guide_tab()

if __name__ == "__main__":
    main()
