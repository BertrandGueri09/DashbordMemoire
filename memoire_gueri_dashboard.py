# memoire_gueri_dashboard.py
# -------------------------------------------------------------
# Dashboard CFAOCI - BRVM (avec fichiers par défaut auto)
# - Charge par défaut (si présents) :
#   /mnt/data/CFAOCI_filtre.csv, /mnt/data/dps_exemple.csv,
#   /mnt/data/eps_exemple.csv, /mnt/data/net_income_exemple.csv
# - L'utilisateur peut importer d'autres fichiers qui écrasent les valeurs par défaut.
# - Analyse technique (fréquences, chandelles, indicateurs)
# - Backtests (SMA, RSI+MACD, Mixte)
# - Fondamentaux de marché dynamiques (plage annuelle calculée depuis les prix)
# - Intégration DPS/EPS/Net Income (fichiers ou saisie manuelle) → Dividend Yield, Dividendes totaux, PER
# - Graphes fondamentaux + Graphe Dividend Yield & PER
# -------------------------------------------------------------

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
    page_title="Dashboard CFAOCI - BRVM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_SHARES_OUTSTANDING = 181_371_900  # modifiable dans la sidebar

# Emplacements par défaut
DEFAULT_PRICE_PATH = "CFAOCI_filtre.csv"
DEFAULT_DPS_PATH = "dps_exemple.csv"
DEFAULT_EPS_PATH = "eps_exemple.csv"
DEFAULT_NET_PATH = "net_income_exemple.csv"

# --------------------------- HELPERS ROBUSTES ---------------------------
def _detect_year_column(df: pd.DataFrame) -> Optional[str]:
    """Détecte la colonne année dans df ('Annee', 'Année', 'Year', 'year', 'period' ou index)."""
    if df is None or df.empty:
        return None
    candidates = ['Annee', 'Année', 'Year', 'year', 'period']
    for c in candidates:
        if c in df.columns:
            return c
    if isinstance(df.index, (pd.Int64Index, pd.UInt64Index, pd.RangeIndex)):
        df['Annee'] = df.index.astype(int)
        return 'Annee'
    for c in df.columns:
        s = df[c]
        try:
            vals = pd.to_numeric(s, errors='coerce')
            if vals.notna().mean() > 0.9:
                if (vals >= 1900).mean() > 0.8 and (vals <= 2100).mean() > 0.8:
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
        y_vals = pd.to_numeric(df[col], errors='coerce')
        y_min = int(np.nanmin(y_vals.values))
        y_max = int(np.nanmax(y_vals.values))
        return (y_min, y_max)
    except Exception:
        return None

# --------------------------- I/O & PARSING ---------------------------
@st.cache_data
def load_data(path_or_buffer: Union[str, io.BytesIO]) -> pd.DataFrame:
    """Charger des prix BRVM (CSV) avec parsing robuste FR (virgules, espaces, K/M)."""
    df = pd.read_csv(path_or_buffer)
    df.columns = df.columns.str.strip()

    rename_map = {
        'Dernier': 'Close', 'Ouv.': 'Open', 'Ouv': 'Open',
        'Plus Haut': 'High', 'Plus Bas': 'Low',
        'Vol.': 'Volume', 'Variation %': 'Variation'
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

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
        val = float(m.group(1))
        suf = m.group(3).lower()
        if suf == 'k': val *= 1_000
        if suf == 'm': val *= 1_000_000
        return val

    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].apply(parse_volume)
    else:
        df['Volume'] = 0.0

    if 'Variation' in df.columns:
        df['Variation'] = df['Variation'].apply(to_num)

    need = ['Date','Close','Open','High','Low']
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Colonne requise manquante: {c}")
    df = df.dropna(subset=need).sort_values('Date').reset_index(drop=True)
    df = df[df['High'] >= df['Low']]
    df = df[(df['High'] >= df['Open']) & (df['High'] >= df['Close'])]
    df = df[(df['Low'] <= df['Open']) & (df['Low'] <= df['Close'])]
    return df

def resample_ohlcv(df: pd.DataFrame, freq_code: str) -> pd.DataFrame:
    dfi = df.set_index('Date')
    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'Variation': 'mean'}
    out = dfi.resample(freq_code).agg(agg).dropna(subset=['Open','High','Low','Close']).reset_index()
    return out

# --------------------------- INDICATEURS TECH ---------------------------
def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window, min_periods=1).mean()

def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
    return prices.ewm(span=window, adjust=False, min_periods=1).mean()

def calculate_rsi(prices: pd.Series, window: int = 14, method: str = "wilder") -> pd.Series:
    delta = prices.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    if method == "wilder":
        roll_up = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        roll_down = down.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    else:
        roll_up = up.rolling(window, min_periods=1).mean()
        roll_down = down.rolling(window, min_periods=1).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.clip(0, 100).fillna(50)

def bollinger_bands(prices: pd.Series, window: int = 20, n_std: float = 2.0):
    ma = prices.rolling(window=window, min_periods=1).mean()
    sd = prices.rolling(window=window, min_periods=1).std(ddof=0)
    return ma - n_std*sd, ma, ma + n_std*sd

def macd(prices: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def _annualization_factor(freq_code: str) -> float:
    return {'D': 252.0, 'W': 52.0, 'M': 12.0}.get(freq_code, 252.0)

def performance_metrics(df: pd.DataFrame, rf_annual_pct: float = 0.0, freq_code: str = 'D') -> Dict[str, float | str]:
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

def plotly_combined_chart(df: pd.DataFrame, chart_type: str, params: Dict) -> go.Figure:
    rows = 1 + int(params.get('show_rsi')) + int(params.get('show_macd'))
    row_heights = [1.0] if rows==1 else ([0.7, 0.3] if rows==2 else [0.6, 0.2, 0.2])
    titles = ['Prix & Volume'] + (['RSI'] if params.get('show_rsi') else []) + (['MACD'] if params.get('show_macd') else [])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights, subplot_titles=titles)

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
        fig.add_shape(type="line", xref=f"x{current_row}", yref=f"y{current_row}",
                      x0=df['Date'].min(), x1=df['Date'].max(), y0=70, y1=70, line=dict(dash="dash", width=1, color="red"))
        fig.add_shape(type="line", xref=f"x{current_row}", yref=f"y{current_row}",
                      x0=df['Date'].min(), x1=df['Date'].max(), y0=50, y1=50, line=dict(dash="dot", width=1, color="gray"))
        fig.add_shape(type="line", xref=f"x{current_row}", yref=f"y{current_row}",
                      x0=df['Date'].min(), x1=df['Date'].max(), y0=30, y1=30, line=dict(dash="dash", width=1, color="green"))
        current_row += 1
    if params.get('show_macd'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_L'], name='MACD', mode='lines'), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_S'], name='Signal', mode='lines'), row=current_row, col=1)
        fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_H'], name='Hist', opacity=0.6), row=current_row, col=1)

    fig.update_layout(height=600, hovermode='x unified', showlegend=True,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                      margin=dict(t=40, b=40, l=40, r=40))
    return fig

# --------------------------- BACKTESTS ---------------------------
def backtest_sma(df: pd.DataFrame, fast: int = 20, slow: int = 50, fee_bps: float = 10.0, cash0: float = 1_000_000.0):
    data = df[['Date','Close']].copy()
    data['SMA_fast'] = calculate_sma(data['Close'], fast)
    data['SMA_slow'] = calculate_sma(data['Close'], slow)
    data['signal'] = (data['SMA_fast'] > data['SMA_slow']).astype(int)
    data['signal_shift'] = data['signal'].shift(1).fillna(0).astype(int)
    data['trade'] = data['signal'] - data['signal_shift']

    fee = fee_bps / 10_000.0
    position, cash, shares = 0, cash0, 0.0
    equity_list, trades = [], []

    for _, row in data.iterrows():
        px = row['Close']
        if row['trade'] == 1 and position == 0:
            shares = (cash * (1 - fee)) / px
            cash, position = 0.0, 1
            trades.append((row['Date'], 'BUY', px, shares))
        elif row['trade'] == -1 and position == 1:
            cash = shares * px * (1 - fee)
            shares, position = 0.0, 0
            trades.append((row['Date'], 'SELL', px, 0.0))
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
        'perf_annualisee_%': ann_ret,
        'vol_annualisee_%': ann_vol,
        'sharpe': sharpe,
        'max_drawdown_%': mdd,
        'nb_trades': int((data['trade'] != 0).sum() // 2)
    }
    trades_df = pd.DataFrame(trades, columns=['Date','Action','Prix','Quantite'])
    return data, stats, trades_df

def backtest_rsi_macd(df: pd.DataFrame, rsi_window=14, rsi_buy=30.0, rsi_confirm=50.0, rsi_sell=70.0,
                      macd_fast=12, macd_slow=26, macd_signal=9, fee_bps=10.0, cash0=1_000_000.0):
    data = df[['Date','Close']].copy()
    data['RSI'] = calculate_rsi(data['Close'], rsi_window, method="wilder")
    macd_l, macd_s, _ = macd(data['Close'], macd_fast, macd_slow, macd_signal)
    data['MACD_L'], data['MACD_S'] = macd_l, macd_s

    prep_flag, prep_list = False, []
    for r in data.itertuples(index=False):
        if r.RSI < rsi_buy:
            prep_flag = True
            prep_list.append(0)
        elif r.RSI >= rsi_confirm and prep_flag:
            prep_list.append(1)
            prep_flag = False
        else:
            prep_list.append(0)
    data['prep'] = prep_list

    data['macd_cross_up'] = ((data['MACD_L'].shift(1) <= data['MACD_S'].shift(1)) & (data['MACD_L'] > data['MACD_S'])).astype(int)
    data['macd_cross_down'] = ((data['MACD_L'].shift(1) >= data['MACD_S'].shift(1)) & (data['MACD_L'] < data['MACD_S'])).astype(int)

    data['buy_signal'] = ((data['prep'] == 1) & (data['macd_cross_up'] == 1)).astype(int)
    data['sell_signal'] = ((data['RSI'] > rsi_sell) | (data['macd_cross_down'] == 1)).astype(int)

    fee = fee_bps / 10_000.0
    position, cash, shares = 0, cash0, 0.0
    equity_list, trades = [], []

    for _, row in data.iterrows():
        px = row['Close']
        if position == 0 and row['buy_signal'] == 1:
            shares = (cash * (1 - fee)) / px
            cash, position = 0.0, 1
            trades.append((row['Date'], 'BUY', px, shares))
        elif position == 1 and row['sell_signal'] == 1:
            cash = shares * px * (1 - fee)
            shares, position = 0.0, 0
            trades.append((row['Date'], 'SELL', px, 0.0))
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
        'perf_annualisee_%': ann_ret,
        'vol_annualisee_%': ann_vol,
        'sharpe': sharpe,
        'max_drawdown_%': mdd,
        'nb_trades': int((data['buy_signal'] == 1).sum())
    }
    trades_df = pd.DataFrame(trades, columns=['Date','Action','Prix','Quantite'])
    return data, stats, trades_df

def backtest_mixed_sma_rsi(df: pd.DataFrame, sma_fast=20, sma_slow=50, rsi_window=14, rsi_enter=55.0, rsi_exit=45.0,
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

    for _, row in data.iterrows():
        px = row['Close']
        if position == 0 and row['enter'] == 1:
            shares = (cash * (1 - fee)) / px
            cash, position = 0.0, 1
            trades.append((row['Date'], 'BUY', px, shares))
        elif position == 1 and row['exit'] == 1:
            cash = shares * px * (1 - fee)
            shares, position = 0.0, 0
            trades.append((row['Date'], 'SELL', px, 0.0))
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
        'perf_annualisee_%': ann_ret,
        'vol_annualisee_%': ann_vol,
        'sharpe': sharpe,
        'max_drawdown_%': mdd,
        'nb_trades': int(((data['enter'] == 1) & (data['exit'].shift(-1) == 1)).sum())
    }
    trades_df = pd.DataFrame(trades, columns=['Date','Action','Prix','Quantite'])
    return data, stats, trades_df

# --------------------------- FONDAMENTAUX (AUTO) ---------------------------
@st.cache_data
def compute_market_fundamentals_from_original(df_original_daily: pd.DataFrame, shares_outstanding: int) -> pd.DataFrame:
    """Calcule fondamentaux annuels (prix fin d’année, rendement, vol annualisée, volume, MDD, capi)."""
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
        cummax = c.cummax()
        return (c/cummax - 1.0).min() * 100
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
    """Lit un CSV avec colonne année + colonne valeur (ex: DPS / EPS / net_income)."""
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
            candidate = c
            break
    if candidate is None:
        cols_lower = {c.lower(): c for c in df.columns}
        for c in value_cols_candidates:
            if c.lower() in cols_lower:
                candidate = cols_lower[c.lower()]
                break
    if candidate is None:
        return None

    out = df[[year_col, candidate]].copy()
    out.rename(columns={year_col: 'Annee', candidate: candidate}, inplace=True)
    out['Annee'] = pd.to_numeric(out['Annee'], errors='coerce').astype('Int64')
    out[candidate] = pd.to_numeric(out[candidate], errors='coerce')
    out = out.dropna(subset=['Annee'])
    return out

def enrich_with_dividends_eps(ann_df: pd.DataFrame,
                              shares_outstanding: int,
                              dps_df: Optional[pd.DataFrame],
                              eps_or_net_df: Optional[pd.DataFrame],
                              manual_dps: Optional[float],
                              manual_payout_pct: Optional[float]) -> pd.DataFrame:
    """Ajoute DPS, EPS (ou calcule via net_income), Dividend Yield, Dividends Total, PER."""
    if ann_df is None or ann_df.empty:
        return ann_df
    out = ann_df.copy()

    if dps_df is not None and not dps_df.empty:
        val_col = [c for c in dps_df.columns if c.lower() in ['dps','dividend_per_share','dividende','dividendes','dividende_par_action']]
        if val_col:
            dps_df = dps_df.rename(columns={val_col[0]: 'DPS'})
        elif 'DPS' not in dps_df.columns:
            if dps_df.shape[1] == 2:
                other = [c for c in dps_df.columns if c != 'Annee'][0]
                dps_df = dps_df.rename(columns={other: 'DPS'})
            else:
                dps_df = None
        if dps_df is not None:
            out = out.merge(dps_df[['Annee','DPS']], on='Annee', how='left')

    if eps_or_net_df is not None and not eps_or_net_df.empty:
        eps_col = None
        net_col = None
        for c in eps_or_net_df.columns:
            if c.lower() in ['eps', 'benefice_par_action', 'bnpa']:
                eps_col = c
                break
        if eps_col is None:
            for c in eps_or_net_df.columns:
                if c.lower() in ['net_income','resultat_net','rn','benefice','profit']:
                    net_col = c
                    break
        temp = eps_or_net_df.copy()
        if eps_col:
            temp = temp.rename(columns={eps_col: 'EPS'})
        elif net_col:
            temp = temp.rename(columns={net_col: 'net_income'})
        else:
            temp = None

        if temp is not None:
            out = out.merge(temp, on='Annee', how='left')
            if 'EPS' not in out.columns and 'net_income' in out.columns:
                out['EPS'] = out['net_income'] / float(shares_outstanding)

    if 'EPS' not in out.columns:
        out['EPS'] = np.nan

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
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['Capitalisation (fin d’année)', 'Rendement annuel (%)',
                                        'Volatilité annualisée (%)', 'Volume annuel (titres)'],
                        vertical_spacing=0.20, horizontal_spacing=0.12)
    fig.add_trace(go.Bar(x=x, y=ann_df['market_cap_fin_annee_FCFA'], name='Capi fin année'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=ann_df['annual_return_%'], name='Rendement annuel', mode='lines+markers'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=ann_df['vol_annual_%'], name='Vol annualisée', mode='lines+markers'), row=2, col=1)
    fig.add_trace(go.Bar(x=x, y=ann_df['vol_sum'], name='Volume annuel'), row=2, col=2)
    fig.update_yaxes(title_text="FCFA", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=2)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="Titres", row=2, col=2)
    fig.update_layout(height=520, showlegend=False, margin=dict(t=90, b=60, l=60, r=60))
    return fig

def plot_dividend_and_pe(ann_df: pd.DataFrame) -> Optional[go.Figure]:
    if ann_df is None or ann_df.empty:
        return None
    year_col = _detect_year_column(ann_df) or 'Annee'
    x = ann_df[year_col]
    has_yield = 'Dividend_Yield_%' in ann_df.columns and ann_df['Dividend_Yield_%'].notna().any()
    has_per = 'PER' in ann_df.columns and ann_df['PER'].notna().any()
    if not has_yield and not has_per:
        return None
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Dividend Yield (%)', 'PER (x)'],
                        shared_xaxes=False, vertical_spacing=0.10, horizontal_spacing=0.12)
    if has_yield:
        fig.add_trace(go.Scatter(x=x, y=ann_df['Dividend_Yield_%'], mode='lines+markers', name='Dividend Yield (%)'), row=1, col=1)
        fig.update_yaxes(title_text="%", row=1, col=1)
    if has_per:
        fig.add_trace(go.Scatter(x=x, y=ann_df['PER'], mode='lines+markers', name='PER (x)'), row=1, col=2)
        fig.update_yaxes(title_text="x", row=1, col=2)
    fig.update_xaxes(title_text="Année", row=1, col=1)
    fig.update_xaxes(title_text="Année", row=1, col=2)
    fig.update_layout(height=360, showlegend=False, margin=dict(t=70, b=50, l=60, r=60))
    return fig

def summarize_fundamentals(ann_df: pd.DataFrame) -> str:
    if ann_df is None or ann_df.empty:
        return "Aucun indicateur fondamental calculable sur la période importée."
    year_col = _detect_year_column(ann_df) or 'Annee'
    ann_df = ann_df.sort_values(year_col).reset_index(drop=True)
    last = ann_df.iloc[-1]
    last_year = int(last[year_col])
    last_price = float(last['last_price'])
    last_cap = int(last['market_cap_fin_annee_FCFA']) if pd.notna(last['market_cap_fin_annee_FCFA']) else None
    last_ret = float(last['annual_return_%']) if pd.notna(last['annual_return_%']) else None
    last_vol = float(last['vol_annual_%']) if pd.notna(last['vol_annual_%']) else None
    last_mdd = float(last['max_drawdown_intra_%']) if pd.notna(last['max_drawdown_intra_%']) else None
    vol_mean = ann_df['vol_sum'].dropna()
    vol_mean = float(vol_mean.mean()) if not vol_mean.empty else None
    first = ann_df.iloc[0]
    first_year = int(first[year_col])
    first_price = float(first['last_price'])
    n_years = max(1, last_year - first_year)
    cagr = None
    if first_price > 0:
        cagr = (last_price / first_price) ** (1 / n_years) - 1
    div_yield = ann_df['Dividend_Yield_%'].iloc[-1] if 'Dividend_Yield_%' in ann_df.columns else None
    div_total = ann_df['Dividends_Total_FCFA'].iloc[-1] if 'Dividends_Total_FCFA' in ann_df.columns else None
    per_last = ann_df['PER'].iloc[-1] if 'PER' in ann_df.columns else None
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
    if pd.notna(per_last): lines.append(f"- **PER {last_year}** : {float(per_last):.2f}x")
    lines.append("> Capi = prix fin d’année × actions. EPS : fourni/calculé, ou estimé via DPS & payout ratio.")
    return "\n".join(lines)

# --------------------------- APP ---------------------------
def main():
    st.title("Dashboard Marchés Boursiers - BRVM")
    with st.sidebar:
        st.header("Données prix")
        uploader = st.file_uploader("Importer le CSV de PRIX (ex: CFAOCI.csv)", type=['csv'], key="price_csv")

        # Charger PRIX : d'abord upload, sinon fallback sur DEFAULT_PRICE_PATH
        if uploader is not None:
            df_original = load_data(uploader)
            st.success("Données de prix chargées depuis le fichier importé.")
        else:
            if os.path.exists(DEFAULT_PRICE_PATH):
                df_original = load_data(DEFAULT_PRICE_PATH)
                st.info(f"Données de prix chargées par défaut : {DEFAULT_PRICE_PATH}")
            else:
                st.error("Aucun fichier de prix. Importez un CSV de prix")
                st.stop()

        shares = st.number_input("Actions en circulation (exactes)", min_value=1, value=DEFAULT_SHARES_OUTSTANDING, step=1000)

        st.header("Période & Fréquence (Analyse technique)")
        freq = st.selectbox("Fréquence", ['Jour', 'Semaine', 'Mois'], index=0)
        freq_map = {'Jour': 'D', 'Semaine': 'W', 'Mois': 'M'}
        freq_code = freq_map[freq]
        min_date, max_date = df_original['Date'].min().date(), df_original['Date'].max().date()
        date_range = st.date_input("Fenêtre d'analyse (graphique technique)", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        else:
            start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)
        df_view = df_original[(df_original['Date'] >= start_date) & (df_original['Date'] <= end_date)].copy()
        df = resample_ohlcv(df_view, freq_code=freq_code)

        st.header("Indicateurs techniques")
        indicators = st.multiselect("Sélection", ['MM', 'EMA', 'Bollinger', 'RSI', 'MACD'], default=['MM', 'RSI'])
        with st.expander("Paramètres indicateurs", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                sma1 = st.slider("MM1", 5, 60, 20, 1)
                sma2 = st.slider("MM2", 10, 200, 50, 1)
                ema1 = st.slider("EMA", 5, 60, 20, 1)
                bb_window = st.slider("BB Fenêtre", 10, 60, 20, 1)
            with col2:
                bb_std = st.slider("BB Écart", 1.0, 3.0, 2.0, 0.1)
                rsi_window = st.slider("RSI", 5, 30, 14, 1)
                macd_fast = st.slider("MACD Rapide", 5, 20, 12, 1)
                macd_slow = st.slider("MACD Lent", 20, 40, 26, 1)
        params = {
            'show_sma': 'MM' in indicators, 'sma1': sma1, 'sma2': sma2,
            'show_ema': 'EMA' in indicators, 'ema1': ema1,
            'show_bb': 'Bollinger' in indicators, 'bb_window': bb_window, 'bb_std': bb_std,
            'show_rsi': 'RSI' in indicators, 'rsi_window': rsi_window,
            'show_macd': 'MACD' in indicators, 'macd_fast': macd_fast, 'macd_slow': macd_slow, 'macd_signal': 9
        }

        st.header("Style & Risque")
        chart_type = st.radio("Type de graphique", ['Ligne', 'Chandelles'])
        rf = st.number_input("Taux sans risque (%)", value=2.0, step=0.5)

        st.header("Backtesting")
        strat = st.selectbox("Stratégie", ["SMA Crossover", "RSI + MACD", "Mixte (SMA + RSI)"])
        if strat == "SMA Crossover":
            colb1, colb2, colb3 = st.columns(3)
            with colb1: bt_fast = st.number_input("MM rapide", min_value=2, max_value=200, value=20, step=1)
            with colb2: bt_slow = st.number_input("MM lente", min_value=5, max_value=400, value=50, step=1)
            with colb3: bt_fee = st.number_input("Frais (bps)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
        elif strat == "RSI + MACD":
            colb1, colb2, colb3 = st.columns(3)
            with colb1: bt_rsi_buy = st.slider("RSI sous-achat (entrée possible)", 10, 40, 30, 1)
            with colb2: bt_rsi_confirm = st.slider("RSI confirmation (repasse au-dessus)", 30, 60, 50, 1)
            with colb3: bt_rsi_sell = st.slider("RSI surachat (sortie)", 50, 90, 70, 1)
            colb4, colb5, colb6 = st.columns(3)
            with colb4: bt_macd_fast = st.slider("MACD rapide", 5, 20, 12, 1)
            with colb5: bt_macd_slow = st.slider("MACD lent", 20, 40, 26, 1)
            with colb6: bt_macd_signal = st.slider("MACD signal", 5, 20, 9, 1)
            bt_fee = st.number_input("Frais (bps)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
        else:
            colb1, colb2, colb3 = st.columns(3)
            with colb1: mix_sma_fast = st.number_input("MM rapide", min_value=2, max_value=200, value=20, step=1)
            with colb2: mix_sma_slow = st.number_input("MM lente", min_value=5, max_value=400, value=50, step=1)
            with colb3: mix_fee = st.number_input("Frais (bps)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
            colb4, colb5 = st.columns(2)
            with colb4: mix_rsi_enter = st.slider("RSI entrée", 40, 70, 55, 1)
            with colb5: mix_rsi_exit = st.slider("RSI sortie", 20, 60, 45, 1)

        # --------- Dividendes & Bénéfices (fichiers par défaut + upload) ---------
        st.header("Dividendes & Bénéfices (facultatif)")
        dps_uploader = st.file_uploader("CSV Dividendes par action (DPS) par année", type=['csv'], key="dps_csv")
        eps_uploader = st.file_uploader("CSV EPS (ou Résultat net) par année", type=['csv'], key="eps_csv")
        st.caption("Colonnes attendues : année = Annee/Année/Year/period ; valeur = DPS | EPS | net_income (FCFA).")

        st.subheader("Saisie manuelle (si pas de fichiers)")
        manual_dps = st.number_input("DPS (dernière année) – optionnel", min_value=0.0, value=0.0, step=1.0, help="Dividende par action en FCFA pour la dernière année de la plage.")
        manual_payout = st.number_input("Payout ratio (%) – optionnel", min_value=0.0, max_value=100.0, value=0.0, step=1.0, help="Si renseigné avec DPS, permet d'estimer l'EPS et donc le PER.")

    # ====== TRAITEMENTS ======
    df = add_indicators(resample_ohlcv(df_view, freq_code=freq_code), params)
    metrics = performance_metrics(df, rf_annual_pct=rf, freq_code=freq_code)

    # Fondamentaux annuels depuis PRIX (non filtrés)
    ann_df = compute_market_fundamentals_from_original(df_original, shares)

    # DPS/EPS/Net Income : upload prioritaire, sinon fichiers par défaut s'ils existent
    if dps_uploader is not None:
        dps_df = _parse_year_value_df(dps_uploader, ['DPS','dps','dividend_per_share','dividende','dividendes','dividende_par_action'])
    elif os.path.exists(DEFAULT_DPS_PATH):
        dps_df = _parse_year_value_df(DEFAULT_DPS_PATH, ['DPS','dps','dividend_per_share','dividende','dividendes','dividende_par_action'])
        st.info(f"DPS chargés par défaut : {DEFAULT_DPS_PATH}")
    else:
        dps_df = None

    if eps_uploader is not None:
        eps_or_net_df = _parse_year_value_df(eps_uploader, ['EPS','eps','net_income','resultat_net','rn','benefice','profit'])
    elif os.path.exists(DEFAULT_EPS_PATH):
        eps_or_net_df = _parse_year_value_df(DEFAULT_EPS_PATH, ['EPS','eps','net_income','resultat_net','rn','benefice','profit'])
        st.info(f"EPS chargés par défaut : {DEFAULT_EPS_PATH}")
    elif os.path.exists(DEFAULT_NET_PATH):
        eps_or_net_df = _parse_year_value_df(DEFAULT_NET_PATH, ['EPS','eps','net_income','resultat_net','rn','benefice','profit'])
        st.info(f"Résultat net chargé par défaut : {DEFAULT_NET_PATH}")
    else:
        eps_or_net_df = None

    manual_dps_val = manual_dps if manual_dps > 0 else None
    manual_payout_val = manual_payout if manual_payout > 0 else None

    ann_df = enrich_with_dividends_eps(ann_df, shares, dps_df, eps_or_net_df, manual_dps_val, manual_payout_val)

    span = _year_span(ann_df)
    fund_title_suffix = f"({span[0]}–{span[1]})" if span else "(n/a)"

    # ====== AFFICHAGE ======
    st.subheader("Métriques principales")
    badge = {"D": "Jour", "W": "Semaine", "M": "Mois"}[freq_code]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(f"Prix ({badge})", f"{metrics['current_price']:.0f} FCFA")
    c2.metric("Rendement total", f"{metrics['total_return']:.1f}%")
    c3.metric("Rend. annualisé", f"{metrics['annualized_return']:.1f}%")
    c4.metric("Volatilité", f"{metrics['volatility']:.1f}%")
    c5.metric("Max DD", f"{metrics['max_drawdown']:.1f}%")
    c6.metric("Sharpe", f"{metrics['sharpe']:.2f}")
    st.caption(f"Période affichée (graphique technique) : {df['Date'].min().date()} → {df['Date'].max().date()} | Dernière MAJ: {metrics['last_update']}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Graphique technique")
        fig = plotly_combined_chart(df, chart_type, params)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"filename":"CFAOCI_chart"}})

    with right:
        st.subheader(f"Fondamentaux de marché {fund_title_suffix}")
        if (ann_df is not None) and (not ann_df.empty):
            fund_fig = plot_market_fundamentals_summary(ann_df)
            st.plotly_chart(fund_fig, use_container_width=True, config={"displaylogo": False})

            extra_fig = plot_dividend_and_pe(ann_df)
            if extra_fig is not None:
                st.plotly_chart(extra_fig, use_container_width=True, config={"displaylogo": False})

            st.markdown(summarize_fundamentals(ann_df))

            fname = f"CFAOCI_fondamentaux_de_marche_{span[0]}_{span[1]}.csv" if span else "CFAOCI_fondamentaux_de_marche.csv"
            st.download_button(
                f"Télécharger fondamentaux de marché {fund_title_suffix} (CSV)",
                ann_df.to_csv(index=False).encode('utf-8'),
                file_name=fname,
                mime="text/csv"
            )
        else:
            st.info("Aucun fondamental de marché calculable (fichier vide ou colonnes manquantes).")

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

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("Capital initial", f"{bt_stats['capital_initial']:,.0f} FCFA")
    d2.metric("Capital final", f"{bt_stats['capital_final']:,.0f} FCFA")
    d3.metric("Perf. totale", f"{bt_stats['perf_totale_%']:.1f}%")
    d4.metric("Perf. annualisée", f"{bt_stats['perf_annualisee_%']:.1f}%")
    d5.metric("Max DD", f"{bt_stats['max_drawdown_%']:.1f}%")
    d6.metric("Sharpe", f"{bt_stats['sharpe']:.2f}")

    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=bt_df['Date'], y=bt_df['equity'], mode='lines', name='Équity'))
    eq_fig.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(eq_fig, use_container_width=True, config={"displaylogo": False})

    cdl1, cdl2 = st.columns(2)
    with cdl1:
        st.download_button("Transactions (CSV)", bt_trades.to_csv(index=False).encode('utf-8'),
                           "CFAOCI_backtest_trades.csv", "text/csv")
    with cdl2:
        st.download_button("Équity (CSV)", bt_df[['Date','equity']].to_csv(index=False).encode('utf-8'),
                           "CFAOCI_backtest_equity.csv", "text/csv")

    st.markdown("---")
    st.info(
        "**Dès que vous importez un fichier, il remplace l’équivalent par défaut.** "
    )

if __name__ == "__main__":
    main()

