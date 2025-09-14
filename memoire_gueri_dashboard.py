# memoire_gueri_dashboard.py — 1 colonne + Thème clair/sombre + Exports PNG + Régimes + Titre centré corrigé
# ------------------------------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import os
import warnings
from typing import Union, Dict, Tuple, List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# ===== CSS de base (compact) =====
BASE_CSS = """
<style>
/* Conteneur global plus large + espacements compacts */
.block-container {padding-top: 0.7rem; padding-bottom: 0.7rem; max-width: 1600px;}
section[data-testid="stSidebar"] .block-container {padding-top: 0.5rem; padding-bottom: 0.5rem;}
div[data-testid="stVerticalBlock"] {gap: 0.6rem;}
.element-container:has(.stPlotlyChart) {margin-bottom: 0.4rem;}
/* Métriques compactes */
[data-testid="stMetric"] div {font-size: 0.9rem;}
[data-testid="stMetricValue"] {font-size: 1.2rem !important;}
[data-testid="stMetricDelta"] {font-size: 0.8rem !important;}
/* Textes lisibles */
p, li { line-height: 1.35; font-size: 0.95rem; }
h2, h3, h4 { margin-bottom: 0.25rem; }
hr { margin: 0.5rem 0 0.6rem 0; }
.small-note { font-size: 0.9rem; color: #666; }

/* ======= Titre centré et corrigé (anti-ligatures) ======= */
.app-title {
  text-align: center;
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif;
  font-weight: 800;
  font-size: clamp(26px, 3.2vw, 36px);
  line-height: 1.15;
  letter-spacing: 0.2px;
  margin: 0.2rem 0 0.4rem 0;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  font-variant-ligatures: none;
  font-feature-settings: "liga" 0, "clig" 0, "kern" 1;
}
.app-subtitle {
  text-align: center;
  margin-top: -0.2rem;
  margin-bottom: 0.6rem;
  opacity: 0.85;
  font-size: 0.95rem;
}
</style>
"""

LIGHT_CSS = """
<style>
body, .block-container { background-color: #ffffff; color: #111; }
.app-subtitle { color: #444; }
</style>
"""

DARK_CSS = """
<style>
body, .block-container { background-color: #0e1117; color: #e8e6e3; }
.small-note { color: #c9c7c4; }
.app-subtitle { color: #c9c7c4; }
</style>
"""

# --------------------------- HELPERS THEME & EXPORT ---------------------------
def apply_theme_css(light_theme: bool):
    st.markdown(BASE_CSS, unsafe_allow_html=True)
    st.markdown(LIGHT_CSS if light_theme else DARK_CSS, unsafe_allow_html=True)

def centered_title(main: str, sub: str = ""):
    html = f'<h1 class="app-title">{main}</h1>'
    if sub:
        html += f'<div class="app-subtitle">{sub}</div>'
    st.markdown(html, unsafe_allow_html=True)

def set_fig_template(fig: go.Figure, light_theme: bool):
    if light_theme:
        fig.update_layout(template="plotly",
                          paper_bgcolor="#white", plot_bgcolor="white",
                          font=dict(color="#000000"))
    else:
        fig.update_layout(template="plotly_dark",
                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                          font=dict(color="#e8e6e3"))


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
    vol = std_p * np.sqrt(ann_fac) * 100
    rf_per_period = (rf_annual_pct/100.0) / ann_fac
    sharpe = 0.0 if std_p == 0 else ((mean_p - rf_per_period) / std_p) * np.sqrt(ann_fac)
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

def plotly_combined_chart(df: pd.DataFrame, chart_type: str, params: Dict, light_theme: bool) -> go.Figure:
    rows = 1 + int(params.get('show_rsi')) + int(params.get('show_macd'))
    row_heights = [1.0] if rows==1 else ([0.68, 0.32] if rows==2 else [0.6, 0.22, 0.18])
    titles = ['Prix & Volume'] + (['RSI'] if params.get('show_rsi') else []) + (['MACD'] if params.get('show_macd') else [])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=row_heights, subplot_titles=titles)

    if chart_type == 'Chandelles':
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Cours'), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Prix', mode='lines', line=dict(width=2)), row=1, col=1)

    if params.get('show_sma'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_1'], name=f"MM{params['sma1']}", mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_2'], name=f"MM{params['sma2']}", mode='lines'), row=1, col=1)
    if params.get('show_ema'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_1'], name=f"EMA{params['ema1']}", mode='lines'), row=1, col=1)
    if params.get('show_bb'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_M'], name="BB", mode='lines', line=dict(dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_U'], showlegend=False, mode='lines', line=dict(width=0)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_L'], fill='tonexty', mode='lines', line=dict(width=0), name='BB Zone', opacity=0.1), row=1, col=1)

    current_row = 2
    if params.get('show_rsi'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', mode='lines'), row=current_row, col=1)
        for y, dash, color in [(70,"dash","red"), (50,"dot","gray"), (30,"dash","green")]:
            fig.add_shape(type="line", xref=f"x{current_row}", yref=f"y{current_row}",
                          x0=df['Date'].min(), x1=df['Date'].max(), y0=y, y1=y,
                          line=dict(dash=dash, width=1, color=color))
        current_row += 1
    if params.get('show_macd'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_L'], name='MACD', mode='lines'), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_S'], name='Signal', mode='lines'), row=current_row, col=1)
        fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_H'], name='Hist', opacity=0.6), row=current_row, col=1)

    fig.update_layout(height=620, hovermode='x unified', showlegend=True,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                      margin=dict(t=30, b=30, l=30, r=20))
    set_fig_template(fig, light_theme)
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
        ann_vol = s_bar * np.sqrt(ann_fac) * 100 if s_bar != 0 else 0.0
        sharpe = 0.0 if s_bar == 0 else (r_bar / s_bar) * np.sqrt(ann_fac)
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
        ann_vol = s_bar * np.sqrt(ann_fac) * 100 if s_bar != 0 else 0.0
        sharpe = 0.0 if s_bar == 0 else (r_bar / s_bar) * np.sqrt(ann_fac)
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
        ann_vol = s_bar * np.sqrt(ann_fac) * 100 if s_bar != 0 else 0.0
        sharpe = 0.0 if s_bar == 0 else (r_bar / s_bar) * np.sqrt(ann_fac)
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

# --------------------------- FONDAMENTAUX AUTO ---------------------------
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
def compute_market_fundamentals_from_original(df_original_daily: pd.DataFrame, shares_outstanding: int) -> pd.DataFrame:
    if df_original_daily.empty:
        return pd.DataFrame()
    df = df_original_daily.copy().sort_values('Date').set_index('Date')
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

def _parse_year_value_df(uploaded: io.BytesIO, value_cols_candidates: List[str]) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(uploaded)
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

    # Estimation via DPS + payout (dernière année)
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

def plot_market_fundamentals_summary(ann_df: pd.DataFrame, light_theme: bool) -> go.Figure:
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
    set_fig_template(fig, light_theme)
    return fig

def plot_dividend_and_pe(ann_df: pd.DataFrame, light_theme: bool) -> Optional[go.Figure]:
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
        fig.add_trace(go.Scatter(x=x, y=ann_df['Dividend_Yield_%'], mode='lines+markers', name='Dividend Yield (%)'), row=1, col=1)
        fig.update_yaxes(title_text="%", row=1, col=1)
    if has_per:
        fig.add_trace(go.Scatter(x=x, y=ann_df['PER'], mode='lines+markers', name='PER (x)'), row=1, col=2)
        fig.update_yaxes(title_text="x", row=1, col=2)
    fig.update_xaxes(title_text="Année", row=1, col=1)
    fig.update_xaxes(title_text="Année", row=1, col=2)
    fig.update_layout(height=380, showlegend=False, margin=dict(t=26, b=18, l=24, r=10))
    set_fig_template(fig, light_theme)
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
    n_years = max(1, last_year - first_year); cagr = None
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

# --------------------------- RÉGIMES DE MARCHÉ (phrases courtes) ---------------------------
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

# --------------------------- APP ---------------------------
def main():
    # ======= Thème depuis la sidebar (appliqué avant tout rendu) =======
    with st.sidebar:
        light_theme = st.toggle("Thème", value=True)
    apply_theme_css(light_theme)

    # ======= Titre centré (corrigé) =======
    centered_title("Dashboard Marchés Boursiers – BRVM",
                   "Analyse technique & fondamentale | Dividend Yield/PE auto | Backtests")

    # ===== SIDEBAR (suite) =====
    with st.sidebar:
        st.header("Données prix")
        uploader = st.file_uploader("Importer le CSV de PRIX", type=['csv'], key="price_csv")
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

        shares = st.number_input("Actions en circulation (exactes)", min_value=1, value=DEFAULT_SHARES_OUTSTANDING, step=1000)

        st.header("Période & Fréquence")
        freq = st.selectbox("Fréquence", ['Jour', 'Semaine', 'Mois'], index=0)
        freq_code = {'Jour':'D','Semaine':'W','Mois':'M'}[freq]
        dmin, dmax = df_original['Date'].min().date(), df_original['Date'].max().date()
        dr = st.date_input("Fenêtre d'analyse (graphique technique)", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        start_date, end_date = (pd.to_datetime(dr[0]), pd.to_datetime(dr[1])) if isinstance(dr, tuple) else (pd.to_datetime(dmin), pd.to_datetime(dmax))
        df_view = df_original[(df_original['Date'] >= start_date) & (df_original['Date'] <= end_date)].copy()

        st.header("Indicateurs techniques")
        indicators = st.multiselect("Sélection", ['MM', 'EMA', 'Bollinger', 'RSI', 'MACD'], default=['MM','RSI'])
        with st.expander("Paramètres", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                sma1 = st.slider("MM1", 5, 60, 20, 1)
                sma2 = st.slider("MM2", 10, 200, 50, 1)
                ema1 = st.slider("EMA", 5, 60, 20, 1)
                bb_window = st.slider("BB Fenêtre", 10, 60, 20, 1)
            with c2:
                bb_std = st.slider("BB Écart", 1.0, 3.0, 2.0, 0.1)
                rsi_window = st.slider("RSI", 5, 30, 14, 1)
                macd_fast = st.slider("MACD Rapide", 5, 20, 12, 1)
                macd_slow = st.slider("MACD Lent", 20, 40, 26, 1)
        params = {
            'show_sma':'MM' in indicators, 'sma1':sma1, 'sma2':sma2,
            'show_ema':'EMA' in indicators, 'ema1':ema1,
            'show_bb':'Bollinger' in indicators, 'bb_window':bb_window, 'bb_std':bb_std,
            'show_rsi':'RSI' in indicators, 'rsi_window':rsi_window,
            'show_macd':'MACD' in indicators, 'macd_fast':macd_fast, 'macd_slow':macd_slow, 'macd_signal':9
        }

        st.header("Style & Risque")
        chart_type = st.radio("Type de graphique", ['Ligne','Chandelles'])
        rf = st.number_input("Taux sans risque (%)", value=2.0, step=0.5)

        st.header("Backtesting")
        strat = st.selectbox("Stratégie", ["SMA Crossover", "RSI + MACD", "Mixte (SMA + RSI)"])
        if strat == "SMA Crossover":
            b1,b2,b3 = st.columns(3)
            with b1: bt_fast = st.number_input("MM rapide", 2, 200, 20, 1)
            with b2: bt_slow = st.number_input("MM lente", 5, 400, 50, 1)
            with b3: bt_fee  = st.number_input("Frais (bps)", 0.0, 200.0, 10.0, 1.0)
        elif strat == "RSI + MACD":
            b1,b2,b3 = st.columns(3)
            with b1: bt_rsi_buy = st.slider("RSI sous-achat", 10, 40, 30, 1)
            with b2: bt_rsi_confirm = st.slider("RSI confirmation", 30, 60, 50, 1)
            with b3: bt_rsi_sell = st.slider("RSI surachat", 50, 90, 70, 1)
            c4,c5,c6 = st.columns(3)
            with c4: bt_macd_fast = st.slider("MACD rapide", 5, 20, 12, 1)
            with c5: bt_macd_slow = st.slider("MACD lent", 20, 40, 26, 1)
            with c6: bt_macd_signal = st.slider("MACD signal", 5, 20, 9, 1)
            bt_fee = st.number_input("Frais (bps)", 0.0, 200.0, 10.0, 1.0)
        else:
            b1,b2,b3 = st.columns(3)
            with b1: mix_sma_fast = st.number_input("MM rapide", 2, 200, 20, 1)
            with b2: mix_sma_slow = st.number_input("MM lente", 5, 400, 50, 1)
            with b3: mix_fee = st.number_input("Frais (bps)", 0.0, 200.0, 10.0, 1.0)
            c4,c5 = st.columns(2)
            with c4: mix_rsi_enter = st.slider("RSI entrée", 40, 70, 55, 1)
            with c5: mix_rsi_exit  = st.slider("RSI sortie", 20, 60, 45, 1)

        # DPS/EPS/Net Income (upload > défaut)
        st.header("Dividendes & Bénéfices (facultatif)")
        dps_uploader = st.file_uploader("CSV DPS par année", type=['csv'], key="dps_csv")
        eps_uploader = st.file_uploader("CSV EPS (ou Résultat net)", type=['csv'], key="eps_csv")
        st.caption("Année = Année ; Valeur = DPS | EPS | net_income (FCFA).")

        st.subheader("Saisie manuelle (si pas de fichiers)")
        manual_dps   = st.number_input("DPS (dernière année)", min_value=0.0, value=0.0, step=1.0)
        manual_payout= st.number_input("Payout ratio (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

    # ===== TRAITEMENTS =====
    df_view = df_original[(df_original['Date'] >= start_date) & (df_original['Date'] <= end_date)].copy()
    df = add_indicators(resample_ohlcv(df_view, freq_code=freq_code), params)
    metrics = performance_metrics(df, rf_annual_pct=rf, freq_code=freq_code)

    ann_df = compute_market_fundamentals_from_original(df_original, shares)

    # DPS / EPS (upload > défauts)
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
    badge = {"D":"Jour","W":"Semaine","M":"Mois"}[freq_code]
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric(f"Prix ({badge})", f"{metrics['current_price']:.0f} FCFA")
    m2.metric("Rendement total", f"{metrics['total_return']:.1f}%")
    m3.metric("Rend. annualisé", f"{metrics['annualized_return']:.1f}%")
    m4.metric("Volatilité", f"{metrics['volatility']:.1f}%")
    m5.metric("Max DD", f"{metrics['max_drawdown']:.1f}%")
    m6.metric("Sharpe", f"{metrics['sharpe']:.2f}")
    st.caption(f"Période affichée : {df['Date'].min().date()} → {df['Date'].max().date()} | Dernière MAJ: {metrics['last_update']}")

    # ===== 1) GRAPHIQUE TECHNIQUE =====
    st.subheader("Graphique technique")
    tech_fig = plotly_combined_chart(df, chart_type, params, light_theme)
    st.plotly_chart(tech_fig, use_container_width=True, config={"displaylogo": False})

    # ===== 2) Dividend Yield & PER (AUTO) =====
    extra_fig = plot_dividend_and_pe(ann_df, light_theme)
    if extra_fig is not None:
        st.subheader("Dividend Yield & PER")
        st.plotly_chart(extra_fig, use_container_width=True, config={"displaylogo": False})

    # ===== 3) Graphiques fondamentaux =====
    st.subheader(f"Fondamentaux de marché {fund_title_suffix}")
    if (ann_df is not None) and (not ann_df.empty):
        fund_fig = plot_market_fundamentals_summary(ann_df, light_theme)
        st.plotly_chart(fund_fig, use_container_width=True, config={"displaylogo": False})

        # ===== 4) Synthèse fondamentale =====
        st.markdown(summarize_fundamentals(ann_df))

        # ===== 5) Régimes de marché =====
        st.markdown("**Régimes de marché (par périodes standards)**")
        for line in describe_market_regimes(ann_df):
            st.write(f"- {line}")

        # ===== 6) Téléchargement fondamentaux =====
        fname = f"CFAOCI_fondamentaux_{span[0]}_{span[1]}.csv" if span else "CFAOCI_fondamentaux.csv"
        st.download_button(
            f"Télécharger fondamentaux {fund_title_suffix} (CSV)",
            ann_df.to_csv(index=False).encode('utf-8'),
            file_name=fname, mime="text/csv"
        )
    else:
        st.info("Aucun fondamental calculable (fichier vide ou colonnes manquantes).")

    # ===== 7) BACKTEST =====
    st.subheader(f"Backtesting — {strat}")
    if strat == "SMA Crossover":
        bt_df, bt_stats, bt_trades = backtest_sma(df, fast=int(bt_fast), slow=int(bt_slow), fee_bps=float(bt_fee))
    elif strat == "RSI + MACD":
        bt_df, bt_stats, bt_trades = backtest_rsi_macd(
            df, rsi_window=int(rsi_window),
            rsi_buy=float(bt_rsi_buy), rsi_confirm=float(bt_rsi_confirm), rsi_sell=float(bt_rsi_sell),
            macd_fast=int(bt_macd_fast), macd_slow=int(bt_macd_slow), macd_signal=int(bt_macd_signal),
            fee_bps=float(bt_fee)
        )
    else:
        bt_df, bt_stats, bt_trades = backtest_mixed_sma_rsi(
            df, sma_fast=int(mix_sma_fast), sma_slow=int(mix_sma_slow),
            rsi_window=int(rsi_window), rsi_enter=float(mix_rsi_enter), rsi_exit=float(mix_rsi_exit),
            fee_bps=float(mix_fee)
        )

    d1,d2,d3,d4,d5,d6 = st.columns(6)
    d1.metric("Capital initial", f"{bt_stats['capital_initial']:,.0f} FCFA")
    d2.metric("Capital final", f"{bt_stats['capital_final']:,.0f} FCFA")
    d3.metric("Perf. totale", f"{bt_stats['perf_totale_%']:.1f}%")
    d4.metric("Perf. annualisée", f"{bt_stats['perf_annualisee_%']:.1f}%")
    d5.metric("Max DD", f"{bt_stats['max_drawdown_%']:.1f}%")
    d6.metric("Sharpe", f"{bt_stats['sharpe']:.2f}")

    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=bt_df['Date'], y=bt_df['equity'], mode='lines', name='Équity'))
    eq_fig.update_layout(height=280, margin=dict(t=6,b=6,l=6,r=6))
    set_fig_template(eq_fig, light_theme)
    st.plotly_chart(eq_fig, use_container_width=True, config={"displaylogo": False})
    
if __name__ == "__main__":
    main()






