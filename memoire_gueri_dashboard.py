# memoire_gueri_dashboard.py
# -------------------------------------------------------------
# Dashboard CFAOCI - BRVM
# - Un SEUL CSV de prix à importer
# - Analyse technique (fréquences dynamiques, chandelles, indicateurs)
# - Backtests: SMA, RSI+MACD, Mixte (SMA+RSI)
# - Fondamentaux de marché recalculés AUTOMATIQUEMENT sur la plage d'années
#   effectivement présente dans le CSV importé (pas de bornes fixes)
# - Accès robuste à la colonne des années (Annee/Année/Year/index)
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
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

# --------------------------- HELPERS ROBUSTES ---------------------------
def _detect_year_column(df: pd.DataFrame) -> Optional[str]:
    """
    Détecte la colonne année dans df parmi :
    'Annee', 'Année', 'Year', 'year' ou un index numérique.
    Si rien trouvé, renvoie None.
    """
    if df is None or df.empty:
        return None

    candidates = ['Annee', 'Année', 'Year', 'year']
    for c in candidates:
        if c in df.columns:
            return c

    # Si aucune colonne explicite — vérifier si l'index est numérique (années)
    if isinstance(df.index, (pd.Int64Index, pd.UInt64Index, pd.RangeIndex)):
        # On crée une colonne 'Annee' à partir de l'index
        df['Annee'] = df.index.astype(int)
        return 'Annee'

    # Dernier recours : si une colonne ressemble à des années (4 chiffres)
    for c in df.columns:
        s = df[c]
        try:
            vals = pd.to_numeric(s, errors='coerce')
            if vals.notna().mean() > 0.9:
                # majoritairement numérique ; tester si ce sont des années plausibles
                if (vals >= 1900).mean() > 0.8 and (vals <= 2100).mean() > 0.8:
                    df.rename(columns={c: 'Annee'}, inplace=True)
                    return 'Annee'
        except Exception:
            continue

    return None

def _year_span(df: pd.DataFrame) -> Optional[Tuple[int, int]]:
    """Retourne (y_min, y_max) si possible, sinon None."""
    if df is None or df.empty:
        return None
    col = _detect_year_column(df)
    if not col:
        return None
    try:
        y_min = int(np.nanmin(pd.to_numeric(df[col], errors='coerce').values))
        y_max = int(np.nanmax(pd.to_numeric(df[col], errors='coerce').values))
        if np.isnan(y_min) or np.isnan(y_max):
            return None
        return (y_min, y_max)
    except Exception:
        return None

# --------------------------- I/O & PARSING ---------------------------
@st.cache_data
def load_data(path_or_buffer: Union[str, io.BytesIO]) -> pd.DataFrame:
    """Charger des prix BRVM (CSV) avec parsing robuste FR (virgules, espaces, K/M)."""
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
        raise ValueError("Colonne 'Date' introuvable.")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])

    # Parse FR -> float
    def to_num(x):
        if pd.isna(x): return np.nan
        s = str(x).replace('\u202f','').replace('\xa0','').replace(' ','').replace(',', '.')
        s = re.sub(r'[^0-9.\-]', '', s)
        return pd.to_numeric(s, errors='coerce')

    for col in ['Close','Open','High','Low']:
        if col in df.columns:
            df[col] = df[col].apply(to_num)

    # Volume (support K/M)
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

    # Nettoyage OHLC
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
    """open=first, high=max, low=min, close=last, volume=sum."""
    dfi = df.set_index('Date')
    agg = {
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'Volume': 'sum', 'Variation': 'mean'
    }
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

    # Prix
    if chart_type == 'Chandelles':
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Cours'), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Prix', mode='lines', line=dict(width=2)), row=1, col=1)

    # MM / EMA
    if params.get('show_sma'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_1'], name=f"MM{params['sma1']}", mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_2'], name=f"MM{params['sma2']}", mode='lines'), row=1, col=1)
    if params.get('show_ema'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_1'], name=f"EMA{params['ema1']}", mode='lines'), row=1, col=1)

    # Bandes
    if params.get('show_bb'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_M'], name="BB", mode='lines', line=dict(dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_U'], showlegend=False, mode='lines', line=dict(width=0)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_L'], fill='tonexty', mode='lines', line=dict(width=0), name='BB Zone', opacity=0.1), row=1, col=1)

    # RSI
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

    # MACD
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
    data['trade'] = data['signal'] - data['signal_shift']  # +1 achat, -1 vente

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

    # Préparation RSI
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

    # MACD cross
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

# --------------------------- FONDAMENTAUX (AUTO, PLAGE DYNAMIQUE) ---------------------------
@st.cache_data
def compute_market_fundamentals_from_original(df_original_daily: pd.DataFrame, shares_outstanding: int) -> pd.DataFrame:
    """
    Calcule des fondamentaux de marché sur TOUTES les années présentes dans df_original_daily :
     - prix fin d'année, prix moyen
     - rendement annuel (dernier/dernier-1)
     - volatilité annualisée (√252 sur ret quotidiens)
     - volume annuel
     - max drawdown intra-année
     - capitalisation fin d'année = prix fin d'année × actions
    Plage d'années déduite automatiquement (min → max du fichier).
    """
    if df_original_daily.empty:
        return pd.DataFrame()

    df = df_original_daily.copy().sort_values('Date').set_index('Date')
    df['ret'] = df['Close'].pct_change()

    ann = df.resample('Y').agg(
        last_price=('Close','last'),
        avg_price=('Close','mean'),
        vol_sum=('Volume','sum')
    )
    # Rendement annuel
    ann['last_close_prev'] = ann['last_price'].shift(1)
    ann['annual_return_%'] = ((ann['last_price']/ann['last_close_prev']) - 1.0) * 100

    # Vol annualisée par année
    def annual_vol(g):
        r = g['ret'].dropna()
        return (r.std() * np.sqrt(252) * 100) if len(r) > 1 else np.nan
    ann['vol_annual_%'] = df.groupby(pd.Grouper(freq='Y')).apply(annual_vol).values

    # Max drawdown intra-année
    def max_dd(g):
        c = g['Close'].dropna()
        if c.empty: return np.nan
        cummax = c.cummax()
        return (c/cummax - 1.0).min() * 100
    ann['max_drawdown_intra_%'] = df.groupby(pd.Grouper(freq='Y')).apply(max_dd).values

    # Capi fin d'année
    ann['market_cap_fin_annee_FCFA'] = ann['last_price'] * float(shares_outstanding)

    # ---------- CRÉATION ROBUSTE DE LA COLONNE 'Annee' ----------
    # Toujours fabriquer Annee à partir de l'index Datetime
    years = ann.index.year.astype(int)
    ann = ann.reset_index(drop=True)
    ann.insert(0, 'Annee', years)

    # Arrondis & types
    ann['last_price'] = ann['last_price'].round(2)
    ann['avg_price'] = ann['avg_price'].round(2)
    ann['vol_sum'] = ann['vol_sum'].round(0).astype('Int64')
    ann['annual_return_%'] = ann['annual_return_%'].round(2)
    ann['vol_annual_%'] = ann['vol_annual_%'].round(2)
    ann['market_cap_fin_annee_FCFA'] = ann['market_cap_fin_annee_FCFA'].round(0).astype('Int64')
    ann['max_drawdown_intra_%'] = ann['max_drawdown_intra_%'].round(2)

    return ann

def plot_market_fundamentals_summary(ann_df: pd.DataFrame) -> go.Figure:
    """4 volets : Capitalisation fin d’année, Rendement annuel, Vol annualisée, Volume annuel."""
    year_col = _detect_year_column(ann_df) or 'Annee'
    x = ann_df[year_col]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Capitalisation (fin d’année)', 'Rendement annuel (%)', 'Volatilité annualisée (%)', 'Volume annuel (titres)'],
        vertical_spacing=0.20, horizontal_spacing=0.12
    )
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
    st.caption("Un seul CSV de prix → Analyse technique & Fondamentaux recalculés automatiquement sur la plage d’années du fichier")

    with st.sidebar:
        st.header("Données")
        uploader = st.file_uploader("Importer le CSV de PRIX (ex: CFAOCI.csv)", type=['csv'], key="price_csv")
        if uploader is None:
            st.info("Importez un fichier CSV pour commencer.")
            st.stop()

        # 1) Charger les données ORIGINALES (pour fondamentaux de marché)
        df_original = load_data(uploader)  # <-- réutilisé partout
        shares = st.number_input("Actions en circulation (hypothèse)", min_value=1, value=DEFAULT_SHARES_OUTSTANDING, step=1000)

        # 2) Paramètres d'affichage (analyse technique)
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

        # IMPORTANT : le graphique technique utilise un sous-ensemble filtré
        df_view = df_original[(df_original['Date'] >= start_date) & (df_original['Date'] <= end_date)].copy()
        df = resample_ohlcv(df_view, freq_code=freq_code)

        # 3) Indicateurs & style
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

    # ====== TRAITEMENTS ======
    # a) Analyse technique (FILTRE + RESAMPLE)
    df = add_indicators(resample_ohlcv(df_view, freq_code=freq_code), params)
    metrics = performance_metrics(df, rf_annual_pct=rf, freq_code=freq_code)

    # b) Fondamentaux de marché sur df_original (NON FILTRÉ) — plage d'années dynamique
    ann_df = compute_market_fundamentals_from_original(df_original, shares)

    # Plage dynamique pour le titre (robuste)
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
            st.dataframe(ann_df, use_container_width=True)

            # nom de fichier dynamique seulement si on a la plage
            if span:
                fname = f"CFAOCI_fondamentaux_de_marche_{span[0]}_{span[1]}.csv"
            else:
                fname = "CFAOCI_fondamentaux_de_marche.csv"

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
        "*À chaque nouveau CSV importé, les fondamentaux de marché sont recalculés* "
        "*sur* **toutes** *les années réellement présentes dans le fichier (plage dynamique).* "
        "**Capitalisation** = *prix fin d’année × actions (paramétrable). Les backtests sont indicatifs.*"
    )

if __name__ == "__main__":
    main()
