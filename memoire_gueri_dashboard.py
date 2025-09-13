# memoire_gueri_dashboard_interactif_patched.py
# -------------------------------------------------------------
# Dashboard CFAOCI - BRVM (corrigé et amélioré)
# - Fréquences dynamiques (jour/sem/mois) correctement appliquées
# - Candlesticks avec mèches (OHLC complet)
# - Backtesting SMA (crossover) + RSI/MACD + Mixte (SMA+RSI)
# - Nouvelle section "Fondamentaux de marché (2006–2025)" calculée depuis les PRIX
#   (capitalisation fin d'année, rendement annuel, vol annualisée, volume, max DD)
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import warnings
from typing import Union, Dict, Tuple, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# --------------------------- CONFIG ---------------------------
st.set_page_config(
    page_title="Dashboard CFAOCI - BRVM (patch)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hypothèse (modifiable dans la sidebar si souhaité)
DEFAULT_SHARES_OUTSTANDING = 181_371_900

# --------------------------- UTILITAIRES ---------------------------
@st.cache_data
def load_data(path_or_buffer: Union[str, io.BytesIO]) -> pd.DataFrame:
    """Charger les données de prix BRVM avec parsing robuste (FR: virgules, espaces, K/M)."""
    df = pd.read_csv(path_or_buffer)
    df.columns = df.columns.str.strip()

    # Standardiser noms colonnes
    rename_map = {
        'Dernier': 'Close', 'Ouv.': 'Open', 'Ouv': 'Open',
        'Plus Haut': 'High', 'Plus Bas': 'Low',
        'Vol.': 'Volume', 'Variation %': 'Variation'
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    # Date
    if 'Date' not in df.columns:
        raise ValueError("Colonne 'Date' introuvable dans le CSV.")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])

    # Parser nombre FR
    def to_num(x):
        if pd.isna(x): return np.nan
        s = str(x)
        s = s.replace('\u202f', '').replace('\xa0', '').replace(' ', '')
        s = s.replace(',', '.')
        s = re.sub(r'[^0-9.\-]', '', s)  # retire % etc.
        return pd.to_numeric(s, errors='coerce')

    # Colonnes OHLC
    for col in ['Close', 'Open', 'High', 'Low']:
        if col in df.columns:
            df[col] = df[col].apply(to_num)

    # Volume
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

    # Variation (en %)
    if 'Variation' in df.columns:
        df['Variation'] = df['Variation'].apply(to_num)

    need = ['Date','Close','Open','High','Low']
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Colonne requise manquante: {c}")
    df = df.dropna(subset=need).sort_values('Date').reset_index(drop=True)

    # Vérification basique de qualité OHLC
    df = df[df['High'] >= df['Low']]
    df = df[(df['High'] >= df['Open']) & (df['High'] >= df['Close'])]
    df = df[(df['Low'] <= df['Open']) & (df['Low'] <= df['Close'])]

    return df

def resample_ohlcv(df: pd.DataFrame, freq_code: str) -> pd.DataFrame:
    """Agrégation OHLCV standard : open=first, high=max, low=min, close=last, vol=sum."""
    dfi = df.set_index('Date')
    agg = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Variation': 'mean'
    }
    out = dfi.resample(freq_code).agg(agg).dropna(subset=['Open','High','Low','Close'])
    out = out.reset_index()
    return out

def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window, min_periods=1).mean()

def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
    return prices.ewm(span=window, adjust=False, min_periods=1).mean()

def calculate_rsi(prices: pd.Series, window: int = 14, method: str = "wilder") -> pd.Series:
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    if method == "wilder":
        roll_up = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        roll_down = down.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    else:
        roll_up = up.rolling(window, min_periods=1).mean()
        roll_down = down.rolling(window, min_periods=1).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.clip(0, 100).fillna(50)

def bollinger_bands(prices: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = prices.rolling(window=window, min_periods=1).mean()
    sd = prices.rolling(window=window, min_periods=1).std(ddof=0)
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    return lower, ma, upper

def macd(prices: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
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

    mean_p = ret.mean()
    std_p = ret.std()
    ann_return = ((1 + mean_p) ** ann_fac - 1) * 100
    vol = std_p * np.sqrt(ann_fac) * 100

    rf_per_period = (rf_annual_pct/100.0) / ann_fac
    sharpe = 0.0 if std_p == 0 else ((mean_p - rf_per_period) / std_p) * np.sqrt(ann_fac)

    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    dd = cum/peak - 1
    max_dd = dd.min() * 100

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
    if rows == 1:
        row_heights = [1.0]
    elif rows == 2:
        row_heights = [0.7, 0.3]
    else:
        row_heights = [0.6, 0.2, 0.2]

    titles = ['Prix & Volume']
    if params.get('show_rsi'): titles.append('RSI')
    if params.get('show_macd'): titles.append('MACD')

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=row_heights, subplot_titles=titles
    )

    if chart_type == 'Chandelles':
        fig.add_trace(
            go.Candlestick(
                x=df['Date'], open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Cours'
            ), row=1, col=1
        )
    else:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Prix', mode='lines', line=dict(width=2)), row=1, col=1)

    if params.get('show_sma'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_1'], name=f"MM{params['sma1']}", mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_2'], name=f"MM{params['sma2']}", mode='lines'), row=1, col=1)
    if params.get('show_ema'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_1'], name=f"EMA{params['ema1']}", mode='lines'), row=1, col=1)

    if params.get('show_bb'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_M'], name="BB", mode='lines', line=dict(dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_U'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_L'], fill='tonexty', mode='lines', line=dict(width=0), name='BB Zone', opacity=0.1), row=1, col=1)

    current_row = 2

    if params.get('show_rsi'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', mode='lines'), row=current_row, col=1)
        fig.add_shape(type="line", xref=f"x{current_row}", yref=f"y{current_row}",
                      x0=df['Date'].min(), x1=df['Date'].max(), y0=70, y1=70,
                      line=dict(dash="dash", width=1, color="red"))
        fig.add_shape(type="line", xref=f"x{current_row}", yref=f"y{current_row}",
                      x0=df['Date'].min(), x1=df['Date'].max(), y0=50, y1=50,
                      line=dict(dash="dot", width=1, color="gray"))
        fig.add_shape(type="line", xref=f"x{current_row}", yref=f"y{current_row}",
                      x0=df['Date'].min(), x1=df['Date'].max(), y0=30, y1=30,
                      line=dict(dash="dash", width=1, color="green"))
        current_row += 1

    if params.get('show_macd'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_L'], name='MACD', mode='lines'), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_S'], name='Signal', mode='lines'), row=current_row, col=1)
        fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_H'], name='Hist', opacity=0.6), row=current_row, col=1)

    fig.update_layout(
        height=600, hovermode='x unified', showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(t=40, b=40, l=40, r=40)
    )
    return fig

# --------------------------- BACKTESTING ---------------------------
def backtest_sma(df: pd.DataFrame, fast: int = 20, slow: int = 50, fee_bps: float = 10.0, cash0: float = 1_000_000.0) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    data = df[['Date','Close']].copy()
    data['SMA_fast'] = calculate_sma(data['Close'], fast)
    data['SMA_slow'] = calculate_sma(data['Close'], slow)
    data['signal'] = (data['SMA_fast'] > data['SMA_slow']).astype(int)
    data['signal_shift'] = data['signal'].shift(1).fillna(0).astype(int)
    data['trade'] = data['signal'] - data['signal_shift']  # +1 achat, -1 vente
    fee = fee_bps / 10_000.0

    position = 0
    cash = cash0
    shares = 0.0
    equity_list = []
    trades = []

    for _, row in data.iterrows():
        price = row['Close']
        if row['trade'] == 1 and position == 0:
            shares = (cash * (1 - fee)) / price
            cash = 0.0
            position = 1
            trades.append((row['Date'], 'BUY', price, shares))
        elif row['trade'] == -1 and position == 1:
            cash = shares * price * (1 - fee)
            shares = 0.0
            position = 0
            trades.append((row['Date'], 'SELL', price, 0.0))
        equity = cash + shares * price
        equity_list.append(equity)

    data['equity'] = equity_list
    data['ret'] = data['equity'].pct_change().fillna(0.0)

    if len(data) > 1:
        ann_fac = _annualization_factor('D')
        r_bar = data['ret'].mean()
        s_bar = data['ret'].std()
        ann_ret = ((1 + r_bar) ** ann_fac - 1) * 100 if r_bar != 0 else 0.0
        ann_vol = s_bar * np.sqrt(ann_fac) * 100 if s_bar != 0 else 0.0
        sharpe = 0.0 if s_bar == 0 else (r_bar / s_bar) * np.sqrt(ann_fac)
        cummax = data['equity'].cummax()
        mdd = (data['equity']/cummax - 1).min() * 100
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

def backtest_rsi_macd(
    df: pd.DataFrame,
    rsi_window: int = 14,
    rsi_buy: float = 30.0,
    rsi_confirm: float = 50.0,
    rsi_sell: float = 70.0,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    fee_bps: float = 10.0,
    cash0: float = 1_000_000.0
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    data = df[['Date','Close']].copy()
    data['RSI'] = calculate_rsi(data['Close'], rsi_window, method="wilder")
    macd_l, macd_s, _ = macd(data['Close'], macd_fast, macd_slow, macd_signal)
    data['MACD_L'], data['MACD_S'] = macd_l, macd_s

    data['prep'] = 0
    prep_flag = False
    prep_list = []
    for r in data.itertuples(index=False):
        if r.RSI < rsi_buy:
            prep_flag = True
        elif r.RSI >= rsi_confirm and prep_flag:
            prep_list.append(1)
            prep_flag = False
            continue
        prep_list.append(0)
    data['prep'] = prep_list

    data['macd_cross_up'] = ((data['MACD_L'].shift(1) <= data['MACD_S'].shift(1)) & (data['MACD_L'] > data['MACD_S'])).astype(int)
    data['macd_cross_down'] = ((data['MACD_L'].shift(1) >= data['MACD_S'].shift(1)) & (data['MACD_L'] < data['MACD_S'])).astype(int)

    data['buy_signal'] = ((data['prep'] == 1) & (data['macd_cross_up'] == 1)).astype(int)
    data['sell_signal'] = ((data['RSI'] > rsi_sell) | (data['macd_cross_down'] == 1)).astype(int)

    fee = fee_bps / 10_000.0
    position = 0
    cash = cash0
    shares = 0.0
    equity_list = []
    trades = []

    for _, row in data.iterrows():
        px = row['Close']
        if position == 0 and row['buy_signal'] == 1:
            shares = (cash * (1 - fee)) / px
            cash = 0.0
            position = 1
            trades.append((row['Date'], 'BUY', px, shares))
        elif position == 1 and row['sell_signal'] == 1:
            cash = shares * px * (1 - fee)
            shares = 0.0
            position = 0
            trades.append((row['Date'], 'SELL', px, 0.0))
        equity_list.append(cash + shares * px)

    data['equity'] = equity_list
    data['ret'] = data['equity'].pct_change().fillna(0.0)

    if len(data) > 1:
        ann_fac = _annualization_factor('D')
        r_bar = data['ret'].mean()
        s_bar = data['ret'].std()
        ann_ret = ((1 + r_bar) ** ann_fac - 1) * 100 if r_bar != 0 else 0.0
        ann_vol = s_bar * np.sqrt(ann_fac) * 100 if s_bar != 0 else 0.0
        sharpe = 0.0 if s_bar == 0 else (r_bar / s_bar) * np.sqrt(ann_fac)
        cummax = data['equity'].cummax()
        mdd = (data['equity']/cummax - 1).min() * 100
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

def backtest_mixed_sma_rsi(
    df: pd.DataFrame,
    sma_fast: int = 20,
    sma_slow: int = 50,
    rsi_window: int = 14,
    rsi_enter: float = 55.0,
    rsi_exit: float = 45.0,
    fee_bps: float = 10.0,
    cash0: float = 1_000_000.0
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    data = df[['Date','Close']].copy()
    data['SMA_fast'] = calculate_sma(data['Close'], sma_fast)
    data['SMA_slow'] = calculate_sma(data['Close'], sma_slow)
    data['RSI'] = calculate_rsi(data['Close'], rsi_window, method="wilder")

    data['enter'] = ((data['SMA_fast'] > data['SMA_slow']) & (data['RSI'] > rsi_enter)).astype(int)
    data['exit']  = ((data['SMA_fast'] < data['SMA_slow']) | (data['RSI'] < rsi_exit)).astype(int)

    fee = fee_bps / 10_000.0
    position = 0
    cash = cash0
    shares = 0.0
    equity_list = []
    trades = []

    for _, row in data.iterrows():
        px = row['Close']
        if position == 0 and row['enter'] == 1:
            shares = (cash * (1 - fee)) / px
            cash = 0.0
            position = 1
            trades.append((row['Date'], 'BUY', px, shares))
        elif position == 1 and row['exit'] == 1:
            cash = shares * px * (1 - fee)
            shares = 0.0
            position = 0
            trades.append((row['Date'], 'SELL', px, 0.0))
        equity_list.append(cash + shares * px)

    data['equity'] = equity_list
    data['ret'] = data['equity'].pct_change().fillna(0.0)

    if len(data) > 1:
        ann_fac = _annualization_factor('D')
        r_bar = data['ret'].mean()
        s_bar = data['ret'].std()
        ann_ret = ((1 + r_bar) ** ann_fac - 1) * 100 if r_bar != 0 else 0.0
        ann_vol = s_bar * np.sqrt(ann_fac) * 100 if s_bar != 0 else 0.0
        sharpe = 0.0 if s_bar == 0 else (r_bar / s_bar) * np.sqrt(ann_fac)
        cummax = data['equity'].cummax()
        mdd = (data['equity']/cummax - 1).min() * 100
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

# --------------------------- "FONDAMENTAUX DE MARCHÉ" (2006–2025 depuis PRIX) ---------------------------
@st.cache_data
def compute_market_fundamentals(df_daily: pd.DataFrame, shares_outstanding: int) -> pd.DataFrame:
    """Construit des indicateurs annuels 2006–2025 à partir des PRIX : capi fin d’année, rendement annuel, vol annualisée, volume annuel, max DD intra-année."""
    df = df_daily.copy().sort_values('Date').set_index('Date')
    df['ret'] = df['Close'].pct_change()

    ann = df.resample('Y').agg(
        last_price=('Close','last'),
        avg_price=('Close','mean'),
        vol_sum=('Volume','sum')
    )
    ann['last_close_prev'] = ann['last_price'].shift(1)
    ann['annual_return_%'] = ((ann['last_price'] / ann['last_close_prev']) - 1.0) * 100

    def annual_vol(g):
        r = g['ret'].dropna()
        return (r.std() * np.sqrt(252) * 100) if len(r) > 1 else np.nan
    ann['vol_annual_%'] = df.groupby(pd.Grouper(freq='Y')).apply(annual_vol).values

    def max_dd(g):
        c = g['Close'].dropna()
        if c.empty: return np.nan
        cummax = c.cummax()
        dd = c / cummax - 1.0
        return dd.min() * 100
    ann['max_drawdown_intra_%'] = df.groupby(pd.Grouper(freq='Y')).apply(max_dd).values

    ann['market_cap_fin_annee_FCFA'] = ann['last_price'] * shares_outstanding

    ann.index = ann.index.year
    ann = ann.loc[(ann.index >= 2006) & (ann.index <= 2025)].copy()

    # Arrondis
    ann['last_price'] = ann['last_price'].round(2)
    ann['avg_price'] = ann['avg_price'].round(2)
    ann['vol_sum'] = ann['vol_sum'].round(0).astype('Int64')
    ann['annual_return_%'] = ann['annual_return_%'].round(2)
    ann['vol_annual_%'] = ann['vol_annual_%'].round(2)
    ann['market_cap_fin_annee_FCFA'] = ann['market_cap_fin_annee_FCFA'].round(0).astype('Int64')
    ann['max_drawdown_intra_%'] = ann['max_drawdown_intra_%'].round(2)
    ann = ann.reset_index().rename(columns={'index':'Année', 0:'Année'})
    ann = ann.rename(columns={'index':'Année', 'level_0':'Année', 'Date':'Année'}).rename(columns={'Année':'Annee'})

    return ann

def plot_market_fundamentals_summary(ann_df: pd.DataFrame) -> go.Figure:
    """4 volets : Capitalisation fin d’année, Rendement annuel, Vol annualisée, Volume annuel."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Capitalisation (fin d’année)', 'Rendement annuel (%)', 'Volatilité annualisée (%)', 'Volume annuel (titres)'],
        vertical_spacing=0.20, horizontal_spacing=0.12
    )

    x = ann_df['Annee']

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

# --------------------------- APP ---------------------------
def main():
    st.title("Dashboard CFAOCI - BRVM")
    st.caption("Analyse technique, fondamentale de marché (2006–2025) & backtesting — fréquences dynamiques")

    with st.sidebar:
        st.header("Données & Période")
        uploader = st.file_uploader("CSV Prix CFAOCI", type=['csv'], key="price_csv")
        if uploader is not None:
            df_raw = load_data(uploader)
        else:
            st.info("Importe ton fichier CSV (ex: CFAOCI.csv).")
            st.stop()

        shares = st.number_input("Actions en circulation (hypothèse)", min_value=1, value=DEFAULT_SHARES_OUTSTANDING, step=1000)

        freq = st.selectbox("Fréquence d'affichage", ['Jour', 'Semaine', 'Mois'], index=0)
        freq_map = {'Jour': 'D', 'Semaine': 'W', 'Mois': 'M'}
        freq_code = freq_map[freq]

        min_date, max_date = df_raw['Date'].min().date(), df_raw['Date'].max().date()
        date_range = st.date_input("Fenêtre d'analyse (graphique)", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        else:
            start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

        dff = df_raw[(df_raw['Date'] >= start_date) & (df_raw['Date'] <= end_date)].copy()
        df = resample_ohlcv(dff, freq_code=freq_code)

        st.subheader("Indicateurs techniques")
        indicators = st.multiselect("Sélection", ['MM', 'EMA', 'Bollinger', 'RSI', 'MACD'], default=['MM', 'RSI'])
        with st.expander("Paramètres techniques", expanded=False):
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
            'show_sma': 'MM' in indicators,
            'sma1': sma1, 'sma2': sma2,
            'show_ema': 'EMA' in indicators, 'ema1': ema1,
            'show_bb': 'Bollinger' in indicators, 'bb_window': bb_window, 'bb_std': bb_std,
            'show_rsi': 'RSI' in indicators, 'rsi_window': rsi_window,
            'show_macd': 'MACD' in indicators, 'macd_fast': macd_fast, 'macd_slow': macd_slow, 'macd_signal': 9
        }

        st.subheader("Style & risque")
        chart_type = st.radio("Type de graphique", ['Ligne', 'Chandelles'])
        rf = st.number_input("Taux sans risque (%)", value=2.0, step=0.5)

        st.subheader("Backtesting — paramètres")
        strat = st.selectbox("Stratégie", ["SMA Crossover", "RSI + MACD", "Mixte (SMA + RSI)"])
        if strat == "SMA Crossover":
            colb1, colb2, colb3 = st.columns(3)
            with colb1:
                bt_fast = st.number_input("MM rapide", min_value=2, max_value=200, value=20, step=1)
            with colb2:
                bt_slow = st.number_input("MM lente", min_value=5, max_value=400, value=50, step=1)
            with colb3:
                bt_fee = st.number_input("Frais (bps)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)

        elif strat == "RSI + MACD":
            colb1, colb2, colb3 = st.columns(3)
            with colb1:
                bt_rsi_buy = st.slider("RSI sous-achat (entrée possible)", 10, 40, 30, 1)
            with colb2:
                bt_rsi_confirm = st.slider("RSI confirmation (repasse au-dessus)", 30, 60, 50, 1)
            with colb3:
                bt_rsi_sell = st.slider("RSI surachat (sortie)", 50, 90, 70, 1)
            colb4, colb5, colb6 = st.columns(3)
            with colb4:
                bt_macd_fast = st.slider("MACD rapide", 5, 20, 12, 1)
            with colb5:
                bt_macd_slow = st.slider("MACD lent", 20, 40, 26, 1)
            with colb6:
                bt_macd_signal = st.slider("MACD signal", 5, 20, 9, 1)
            bt_fee = st.number_input("Frais (bps)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)

        else:  # Mixte (SMA + RSI)
            colb1, colb2, colb3 = st.columns(3)
            with colb1:
                mix_sma_fast = st.number_input("MM rapide", min_value=2, max_value=200, value=20, step=1)
            with colb2:
                mix_sma_slow = st.number_input("MM lente", min_value=5, max_value=400, value=50, step=1)
            with colb3:
                mix_fee = st.number_input("Frais (bps)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
            colb4, colb5 = st.columns(2)
            with colb4:
                mix_rsi_enter = st.slider("RSI entrée", 40, 70, 55, 1)
            with colb5:
                mix_rsi_exit = st.slider("RSI sortie", 20, 60, 45, 1)

    # TRAITEMENTS
    df = add_indicators(df, params)
    metrics = performance_metrics(df, rf_annual_pct=rf, freq_code=freq_code)

    # FONDAMENTAUX DE MARCHÉ 2006–2025 (calculés sur TOUT l'historique disponible)
    ann_df = compute_market_fundamentals(df_raw, shares)

    # MÉTRIQUES PRINCIPALES
    st.subheader("Métriques principales")
    badge = {"D": "Jour", "W": "Semaine", "M": "Mois"}[freq_code]
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric(f"Prix ({badge})", f"{metrics['current_price']:.0f} FCFA")
    col2.metric("Rendement total", f"{metrics['total_return']:.1f}%")
    col3.metric("Rend. annualisé", f"{metrics['annualized_return']:.1f}%")
    col4.metric("Volatilité", f"{metrics['volatility']:.1f}%")
    col5.metric("Max DD", f"{metrics['max_drawdown']:.1f}%")
    col6.metric("Sharpe", f"{metrics['sharpe']:.2f}")
    st.caption(f"Période affichée (graphiques prix) : {df['Date'].min().date()} → {df['Date'].max().date()} | Dernière MAJ: {metrics['last_update']}")

    # LAYOUT
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Graphique principal")
        main_fig = plotly_combined_chart(df, chart_type, params)
        st.plotly_chart(main_fig, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"filename": "CFAOCI_chart"}})

    with col_right:
        st.subheader("Fondamentaux de marché (2006–2025)")
        fund_fig = plot_market_fundamentals_summary(ann_df)
        st.plotly_chart(fund_fig, use_container_width=True, config={"displaylogo": False})
        st.dataframe(ann_df, use_container_width=True)

        st.download_button(
            "Télécharger fondamentaux de marché (CSV)",
            ann_df.to_csv(index=False).encode('utf-8'),
            file_name="CFAOCI_fondamentaux_de_marche_2006_2025.csv",
            mime="text/csv"
        )

    # BACKTEST
    st.subheader(f"Backtesting — {strat}")
    if strat == "SMA Crossover":
        bt_df, bt_stats, bt_trades = backtest_sma(df, fast=int(bt_fast), slow=int(bt_slow), fee_bps=float(bt_fee))
    elif strat == "RSI + MACD":
        bt_df, bt_stats, bt_trades = backtest_rsi_macd(
            df,
            rsi_window=int(rsi_window),
            rsi_buy=float(bt_rsi_buy),
            rsi_confirm=float(bt_rsi_confirm),
            rsi_sell=float(bt_rsi_sell),
            macd_fast=int(bt_macd_fast),
            macd_slow=int(bt_macd_slow),
            macd_signal=int(bt_macd_signal),
            fee_bps=float(bt_fee)
        )
    else:
        bt_df, bt_stats, bt_trades = backtest_mixed_sma_rsi(
            df,
            sma_fast=int(mix_sma_fast),
            sma_slow=int(mix_sma_slow),
            rsi_window=int(rsi_window),
            rsi_enter=float(mix_rsi_enter),
            rsi_exit=float(mix_rsi_exit),
            fee_bps=float(mix_fee)
        )

    colbA, colbB, colbC, colbD, colbE, colbF = st.columns(6)
    colbA.metric("Capital initial", f"{bt_stats['capital_initial']:,.0f} FCFA")
    colbB.metric("Capital final", f"{bt_stats['capital_final']:,.0f} FCFA")
    colbC.metric("Perf. totale", f"{bt_stats['perf_totale_%']:.1f}%")
    colbD.metric("Perf. annualisée", f"{bt_stats['perf_annualisee_%']:.1f}%")
    colbE.metric("Max DD", f"{bt_stats['max_drawdown_%']:.1f}%")
    colbF.metric("Sharpe", f"{bt_stats['sharpe']:.2f}")

    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=bt_df['Date'], y=bt_df['equity'], mode='lines', name='Équity'))
    eq_fig.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(eq_fig, use_container_width=True, config={"displaylogo": False})

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("Télécharger les transactions (CSV)", bt_trades.to_csv(index=False).encode('utf-8'),
                           "CFAOCI_backtest_trades.csv", "text/csv")
    with col_dl2:
        st.download_button("Télécharger l'équity (CSV)", bt_df[['Date','equity']].to_csv(index=False).encode('utf-8'),
                           "CFAOCI_backtest_equity.csv", "text/csv")

    st.markdown("---")
    st.info(
        "**Qualité des données** : *vérifiez que votre CSV correspond bien aux cours officiels BRVM.* "
        "*Les fondamentaux de marché (2006–2025) sont dérivés des PRIX (pas des états financiers).* "
        "*Capitalisation = prix de fin d’année × actions en circulation (hypothèse modifiable dans la sidebar).* "
        "*Les backtests sont indicatifs (pas de slippage/gaps hors séance).*"
    )

if __name__ == "__main__":
    main()
