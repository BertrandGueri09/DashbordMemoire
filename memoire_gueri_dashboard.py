import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import io
from typing import Dict, Tuple, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------- CONFIG ---------------------------
st.set_page_config(
    page_title="Dashboard CFAOCI - BRVM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Style CSS pour thème gris clair complet de l'application
st.markdown("""
<style>
    .main > div {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    .stApp {
        background-color: #f5f5f5;
        color: #2c3e50;
    }
    .block-container {
        background-color: #f8f9fa;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        max-width: 100%;
    }
    .stSidebar > div {
        background-color: #e9ecef;
        border-right: 1px solid #ced4da;
    }
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #343a40 !important;
        font-weight: 600;
        background-color: rgba(233, 236, 239, 0.3);
        padding: 0.5rem;
        border-radius: 6px;
        border-left: 4px solid #6c757d;
    }
    div[data-testid="metric-container"] {
        background-color: #e9ecef;
        border: 1px solid #ced4da;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .stPlotlyChart {
        border: 1px solid #ced4da;
        border-radius: 8px;
        padding: 12px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .stButton > button {
        background-color: #6c757d;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# --------------------------- UTILITAIRES TECHNIQUES ---------------------------
@st.cache_data
def load_data(path_or_buffer: str | io.BytesIO) -> pd.DataFrame:
    """Charger et traiter les données CSV (prix) - Version corrigée"""
    try:
        df = pd.read_csv(path_or_buffer)
        df.columns = df.columns.str.strip()
        
        # Essayer plusieurs formats de date
        if 'Date' in df.columns:
            # Format principal: YYYY-MM-DD
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
            # Si échec, essayer DD/MM/YYYY
            if df['Date'].isna().any():
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            # Dernier recours: inférence automatique
            if df['Date'].isna().any():
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Mapping des colonnes
        rename_map = {'Dernier': 'Close', 'Ouv.': 'Open', 'Plus Haut': 'High', 'Plus Bas': 'Low'}
        df = df.rename(columns=rename_map)

        # Conversion des colonnes de prix
        for col in ['Close', 'Open', 'High', 'Low']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Traitement du volume
        def parse_volume(vol_str):
            if pd.isna(vol_str) or str(vol_str).strip() == '':
                return 0.0
            s = str(vol_str).replace(',', '.').replace(' ', '').strip()
            try:
                if s.upper().endswith('K'):
                    return float(s[:-1]) * 1_000
                elif s.upper().endswith('M'):
                    return float(s[:-1]) * 1_000_000
                elif s.upper().endswith('B'):
                    return float(s[:-1]) * 1_000_000_000
                else:
                    return float(s)
            except (ValueError, TypeError):
                return 0.0

        if 'Vol.' in df.columns:
            df['Volume'] = df['Vol.'].apply(parse_volume)
        elif 'Volume' not in df.columns:
            df['Volume'] = 100.0  # Volume par défaut

        # Traitement de la variation
        def parse_variation(var_str):
            if pd.isna(var_str) or str(var_str).strip() == '':
                return 0.0
            try:
                s = str(var_str).replace('%', '').replace(',', '.').strip()
                return float(s)
            except (ValueError, TypeError):
                return 0.0

        if 'Variation %' in df.columns:
            df['Variation'] = df['Variation %'].apply(parse_variation)
        elif 'Variation' not in df.columns:
            df['Variation'] = 0.0

        # Nettoyage et validation
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Validation des données OHLC
        df = validate_and_fix_ohlc(df)
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

def validate_and_fix_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Validation et correction des données OHLC pour garantir la cohérence"""
    if df.empty:
        return df
        
    # Colonnes requises
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"Colonnes manquantes: {missing_cols}")
        return df
    
    # Supprimer les lignes avec des valeurs nulles ou nulles dans OHLC
    df = df.dropna(subset=required_cols)
    df = df[df[required_cols].min(axis=1) > 0]  # Éliminer les prix négatifs ou nuls
    
    # Correction des incohérences OHLC
    for idx in df.index:
        open_price = df.loc[idx, 'Open']
        close_price = df.loc[idx, 'Close'] 
        high_price = df.loc[idx, 'High']
        low_price = df.loc[idx, 'Low']
        
        # Le High doit être au moins égal au max(Open, Close)
        min_high = max(open_price, close_price)
        if high_price < min_high:
            df.loc[idx, 'High'] = min_high
            
        # Le Low doit être au plus égal au min(Open, Close)  
        max_low = min(open_price, close_price)
        if low_price > max_low:
            df.loc[idx, 'Low'] = max_low
    
    return df

def resample_data_by_frequency(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Rééchantillonnage correct des données selon la fréquence - CORRECTION MAJEURE"""
    if df.empty:
        return df
    
    # Mapping des fréquences
    freq_map = {
        'Jour': 'D',
        'Semaine': 'W-MON',  # Semaine commençant le lundi
        'Mois': 'M'  # Fin de mois
    }
    
    pandas_freq = freq_map.get(frequency, 'D')
    
    # Si fréquence = jour, retourner tel quel
    if frequency == 'Jour':
        return df
    
    try:
        # Définir Date comme index
        df_indexed = df.set_index('Date')
        
        # Agrégation OHLC correcte
        ohlc_agg = {
            'Open': 'first',   # Premier open de la période
            'High': 'max',     # Plus haut de la période  
            'Low': 'min',      # Plus bas de la période
            'Close': 'last',   # Dernier close de la période
            'Volume': 'sum',   # Volume cumulé
        }
        
        # Ajouter d'autres colonnes si elles existent
        for col in df_indexed.columns:
            if col not in ohlc_agg:
                if col in ['Variation']:
                    ohlc_agg[col] = 'mean'
                elif df_indexed[col].dtype in ['float64', 'int64']:
                    ohlc_agg[col] = 'last'
        
        # Effectuer le rééchantillonnage
        resampled = df_indexed.resample(pandas_freq).agg(ohlc_agg)
        
        # Supprimer les lignes avec des NaN
        resampled = resampled.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Recalculer la variation pour la nouvelle fréquence
        resampled['Variation'] = resampled['Close'].pct_change().fillna(0) * 100
        
        # Réinitialiser l'index
        result = resampled.reset_index()
        
        return result if not result.empty else df
        
    except Exception as e:
        st.warning(f"Erreur lors du rééchantillonnage: {e}")
        return df

def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window, min_periods=1).mean()

def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
    return prices.ewm(span=window, adjust=False, min_periods=1).mean()

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
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

def performance_metrics(df: pd.DataFrame, rf_annual_pct: float = 0.0) -> Dict[str, float | str]:
    """Métriques de performance corrigées"""
    if df.empty or 'Close' not in df.columns:
        return {}
        
    try:
        latest = df.iloc[-1]
        oldest = df.iloc[0]
        
        # Rendement total corrigé - éviter division par zéro
        if oldest['Close'] > 0:
            total_return = ((latest['Close'] - oldest['Close']) / oldest['Close']) * 100
        else:
            total_return = 0.0
        
        # Rendement annualisé
        n_days = (latest['Date'] - oldest['Date']).days
        if n_days > 0 and oldest['Close'] > 0:
            ann_return = ((latest['Close'] / oldest['Close']) ** (252 / n_days) - 1) * 100
        else:
            ann_return = 0.0

        # Calculs de volatilité et Sharpe
        daily_ret = df['Close'].pct_change().dropna()
        vol = daily_ret.std() * np.sqrt(252) * 100 if len(daily_ret) > 1 else 0.0
        rf_daily = (rf_annual_pct / 100) / 252
        
        if len(daily_ret) > 1 and daily_ret.std() > 0:
            sharpe = ((daily_ret.mean() - rf_daily) / daily_ret.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Drawdown maximum
        cummax = df['Close'].cummax()
        dd = df['Close'] / cummax - 1.0
        max_dd = dd.min() * 100

        return {
            'current_price': float(latest['Close']),
            'total_return': float(total_return),
            'annualized_return': float(ann_return),
            'volatility': float(vol),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_dd),
            'avg_volume': float(df['Volume'].mean()),
            'max_price': float(df['Close'].max()),
            'min_price': float(df['Close'].min()),
            'last_update': latest['Date'].strftime('%d/%m/%Y')
        }
    except Exception as e:
        st.error(f"Erreur dans le calcul des métriques: {e}")
        return {}

def add_indicators(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    if df.empty:
        return df
        
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
        df['RSI'] = calculate_rsi(df['Close'], params['rsi_window'])
    if params.get('show_macd'):
        macd_l, macd_s, macd_h = macd(df['Close'], params['macd_fast'], params['macd_slow'], params['macd_signal'])
        df['MACD_L'], df['MACD_S'], df['MACD_H'] = macd_l, macd_s, macd_h
    return df

def create_advanced_candlestick_chart(df: pd.DataFrame, chart_type: str, params: Dict) -> go.Figure:
    """Graphique en chandelles avancé avec mèches correctes - CORRECTION MAJEURE"""
    if df.empty:
        return go.Figure()
    
    # Calculer le nombre de sous-graphiques nécessaires
    subplot_count = 2  # Prix + Volume par défaut
    if params.get('show_rsi'): subplot_count += 1
    if params.get('show_macd'): subplot_count += 1
    
    # Définir les hauteurs des sous-graphiques
    if subplot_count == 2:
        heights = [0.8, 0.2]
        titles = ['Prix CFAOCI', 'Volume']
    elif subplot_count == 3:
        heights = [0.6, 0.2, 0.2] 
        titles = ['Prix CFAOCI', 'Volume', 'RSI' if params.get('show_rsi') else 'MACD']
    else:
        heights = [0.5, 0.2, 0.15, 0.15]
        titles = ['Prix CFAOCI', 'Volume', 'RSI', 'MACD']
    
    # Créer les sous-graphiques
    fig = make_subplots(
        rows=subplot_count, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=heights,
        subplot_titles=titles
    )
    
    current_row = 1
    
    # === GRAPHIQUE PRINCIPAL (Prix) ===
    if chart_type == 'Chandelles':
        # CHANDELLES CORRECTES avec toutes les mèches
        fig.add_trace(
            go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],  # CORRECTION: Utilisation correcte du High
                low=df['Low'],    # CORRECTION: Utilisation correcte du Low  
                close=df['Close'],
                name='CFAOCI',
                increasing=dict(line=dict(color='#00ff88', width=1), fillcolor='#00ff88'),
                decreasing=dict(line=dict(color='#ff4444', width=1), fillcolor='#ff4444'),
                line=dict(width=1),
            ),
            row=current_row, col=1
        )
    else:
        # Mode ligne
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['Close'], 
                mode='lines', 
                name='Prix de Clôture',
                line=dict(color='#2E86AB', width=2)
            ),
            row=current_row, col=1
        )
    
    # Ajout des indicateurs sur le graphique principal
    if params.get('show_sma'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_1'], name=f"MM {params['sma1']}", 
                                mode='lines', line=dict(width=1.5, dash='solid')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_2'], name=f"MM {params['sma2']}", 
                                mode='lines', line=dict(width=1.5, dash='solid')), row=current_row, col=1)
    
    if params.get('show_ema'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_1'], name=f"EMA {params['ema1']}", 
                                mode='lines', line=dict(width=1.5, dash='dot')), row=current_row, col=1)
    
    if params.get('show_bb'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_M'], name="BB Moyenne", 
                                mode='lines', line=dict(width=1, dash='dash', color='orange')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_U'], mode='lines', 
                                line=dict(width=0), showlegend=False), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_L'], fill='tonexty', mode='lines',
                                line=dict(width=0), name='Bandes Bollinger', opacity=0.1), row=current_row, col=1)
    
    current_row += 1
    
    # === VOLUME ===
    # Couleur du volume selon la direction (vert si hausse, rouge si baisse)
    volume_colors = ['#00ff88' if close >= open else '#ff4444' 
                     for close, open in zip(df['Close'], df['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=df['Date'], 
            y=df['Volume'], 
            name='Volume',
            marker_color=volume_colors,
            opacity=0.7
        ),
        row=current_row, col=1
    )
    current_row += 1
    
    # === RSI ===
    if params.get('show_rsi'):
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', 
                      mode='lines', line=dict(color='purple', width=2)),
            row=current_row, col=1
        )
        # Lignes de référence RSI
        fig.add_hline(y=70, line_dash='dash', line_color='red', opacity=0.7, row=current_row, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='green', opacity=0.7, row=current_row, col=1)
        fig.add_hline(y=50, line_dash='dot', line_color='gray', opacity=0.5, row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        current_row += 1
    
    # === MACD ===
    if params.get('show_macd'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_L'], name='MACD', 
                                mode='lines', line=dict(color='blue', width=2)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_S'], name='Signal', 
                                mode='lines', line=dict(color='red', width=2)), row=current_row, col=1)
        
        # Histogramme MACD
        macd_colors = ['green' if val >= 0 else 'red' for val in df['MACD_H']]
        fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_H'], name='Histogramme MACD',
                            marker_color=macd_colors, opacity=0.6), row=current_row, col=1)
    
    # === CONFIGURATION DU LAYOUT ===
    fig.update_layout(
        title=f'Dashboard CFAOCI - {chart_type}',
        height=700,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        margin=dict(t=80, b=40, l=60, r=60),
        template='plotly_white'
    )
    
    # Mise à jour des axes
    fig.update_yaxes(title_text="Prix (FCFA)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    if params.get('show_rsi'):
        row_rsi = 3 if subplot_count >= 3 else 2
        fig.update_yaxes(title_text="RSI", row=row_rsi, col=1)
    
    if params.get('show_macd'):
        row_macd = subplot_count
        fig.update_yaxes(title_text="MACD", row=row_macd, col=1)
    
    # Suppression du range selector par défaut pour plus d'espace
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    return fig

def advanced_backtest_strategy(df: pd.DataFrame, strategy: str = "sma_crossover") -> Dict:
    """Système de backtest avancé avec stratégies multiples"""
    if len(df) < 100:
        return {"error": "Données insuffisantes (minimum 100 points requis)"}
    
    df = df.copy()
    initial_capital = 1000000  # 1M FCFA
    position_size = 0.02  # 2% du capital par trade
    
    results = {
        "trades": [],
        "equity_curve": [],
        "total_return": 0,
        "win_rate": 0,
        "max_drawdown": 0,
        "sharpe_ratio": 0,
        "total_trades": 0
    }
    
    if strategy == "sma_crossover":
        # Stratégie croisement moyennes mobiles 20/50
        df['SMA_20'] = calculate_sma(df['Close'], 20)
        df['SMA_50'] = calculate_sma(df['Close'], 50)
        
        # Génération des signaux
        df['Signal'] = 0
        df.loc[df['SMA_20'] > df['SMA_50'], 'Signal'] = 1  # Long
        df.loc[df['SMA_20'] < df['SMA_50'], 'Signal'] = -1  # Short
        
        # Détection des changements de signal
        df['Position_Change'] = df['Signal'].diff()
        
        # Simulation des trades
        capital = initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        
        for idx, row in df.iterrows():
            current_date = row['Date']
            current_price = row['Close']
            
            if row['Position_Change'] == 2:  # Signal d'achat (passage de -1 à 1)
                if position == 0:  # Pas de position
                    position = (capital * position_size) / current_price
                    entry_price = current_price
                    entry_date = current_date
                    
            elif row['Position_Change'] == -2:  # Signal de vente (passage de 1 à -1)
                if position > 0:  # Position longue
                    exit_price = current_price
                    pnl = (exit_price - entry_price) * position
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    
                    capital += pnl
                    
                    results["trades"].append({
                        "entry_date": entry_date,
                        "exit_date": current_date,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "position_size": position
                    })
                    
                    position = 0
            
            # Calcul de l'equity curve
            if position > 0:
                current_value = capital + (current_price - entry_price) * position
            else:
                current_value = capital
                
            results["equity_curve"].append({
                "date": current_date,
                "equity": current_value,
                "price": current_price
            })
    
    # Calcul des statistiques finales
    if results["trades"]:
        trades = results["trades"]
        pnl_list = [t["pnl_pct"] for t in trades]
        
        results["total_return"] = (capital - initial_capital) / initial_capital * 100
        results["total_trades"] = len(trades)
        results["win_rate"] = len([p for p in pnl_list if p > 0]) / len(pnl_list) * 100
        results["avg_win"] = np.mean([p for p in pnl_list if p > 0]) if any(p > 0 for p in pnl_list) else 0
        results["avg_loss"] = np.mean([p for p in pnl_list if p < 0]) if any(p < 0 for p in pnl_list) else 0
        
        # Calcul du drawdown maximum
        equity_values = [e["equity"] for e in results["equity_curve"]]
        peak = equity_values[0]
        max_dd = 0
        for equity in equity_values:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        results["max_drawdown"] = max_dd
        
        # Sharpe ratio approximatif
        if len(pnl_list) > 1:
            results["sharpe_ratio"] = np.mean(pnl_list) / np.std(pnl_list) * np.sqrt(252) if np.std(pnl_list) > 0 else 0
    
    return results

# --------------------------- FONDAMENTAUX ---------------------------
@st.cache_data
def load_fundamentals(path_or_buffer: str | io.BytesIO) -> pd.DataFrame:
    """Charger un CSV de fondamentaux (période, CA, RN, etc.)"""
    df = pd.read_csv(path_or_buffer)
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if c != "period":
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def fundamentals_default_df() -> pd.DataFrame:
    """Données fondamentales par défaut avec corrections de cohérence"""
    data = [
        ["2020",  99126, 3780, 181_371_900, np.nan, 22.15, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2021", 119731, 6711, 181_371_900, np.nan, 69.47, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2022", 146375, 5534, 181_371_900, np.nan, 28.67, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2023", 180162, 6399, 181_371_900, np.nan, 15.88, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2024", 158313, 4693, 181_371_900, np.nan,  7.04, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2025", np.nan, np.nan, 181_371_900, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]
    cols = ["period","revenue","net_income","shares_outstanding","dividends_total","dividend_per_share","total_equity","total_debt","total_assets","cash_and_equivalents","capex","EPS"]
    return pd.DataFrame(data, columns=cols)

def _fit_yearly_trend_impute(df: pd.DataFrame, col: str) -> pd.Series:
    """Régression linéaire pour estimer les données manquantes"""
    s = df[['period', col]].dropna()
    out = df[col].copy()
    try:
        years = pd.to_numeric(s['period'], errors='coerce')
        mask = years.notna() & s[col].notna()
        if mask.sum() >= 2:
            x = years[mask].values
            y = s[col][mask].values
            coef = np.polyfit(x, y, 1)
            poly = np.poly1d(coef)
            target_years = pd.to_numeric(df['period'], errors='coerce').values
            pred = poly(target_years)
            out = out.where(out.notna(), pred)
    except Exception:
        pass
    out = out.ffill().bfill()
    return out

def impute_fundamentals(df_fund: pd.DataFrame, assume_roe: float, assume_dte: float, last_close: float) -> pd.DataFrame:
    """Traitement des données fondamentales avec corrections - VERSION CORRIGÉE"""
    df = df_fund.copy()
    df['period'] = df['period'].astype(str)

    # Imputation des données manquantes
    if 'revenue' in df.columns:
        df['revenue'] = _fit_yearly_trend_impute(df, 'revenue')
    if 'net_income' in df.columns:
        df['net_income'] = _fit_yearly_trend_impute(df, 'net_income')
    if 'dividend_per_share' in df.columns:
        df['dividend_per_share'] = df['dividend_per_share'].ffill().bfill()

    # Calcul EPS corrigé
    if 'EPS' not in df.columns:
        df['EPS'] = np.nan
    if {'net_income', 'shares_outstanding'} <= set(df.columns):
        mask = (df['shares_outstanding'] > 0) & df['net_income'].notna()
        df.loc[mask, 'EPS'] = df.loc[mask, 'net_income'] / df.loc[mask, 'shares_outstanding']

    # Calcul PER corrigé - CORRECTION MAJEURE
    df['PER'] = np.nan
    mask = (df['EPS'] > 0.01) & (last_close > 0)  # Éviter les divisions par des valeurs trop petites
    df.loc[mask, 'PER'] = last_close / df.loc[mask, 'EPS']
    # Limiter le PER à des valeurs raisonnables (entre 1 et 200)
    df['PER'] = np.where((df['PER'] < 1) | (df['PER'] > 200), np.nan, df['PER'])

    # Calcul des autres ratios avec validations
    if {'dividend_per_share', 'shares_outstanding'} <= set(df.columns):
        if 'dividends_total' not in df.columns:
            df['dividends_total'] = np.nan
        mask = df['shares_outstanding'] > 0
        df.loc[mask, 'dividends_total'] = df.loc[mask, 'dividend_per_share'] * df.loc[mask, 'shares_outstanding']

    # Capitaux propres estimés
    if 'total_equity' not in df.columns:
        df['total_equity'] = np.nan
    if 'net_income' in df.columns:
        roe = max(assume_roe, 0.001)  # ROE minimum de 0.1%
        mask = (df['net_income'] > 0) & df['total_equity'].isna()
        df.loc[mask, 'total_equity'] = df.loc[mask, 'net_income'] / roe

    # Dette totale estimée
    if 'total_debt' not in df.columns:
        df['total_debt'] = np.nan
    mask = df['total_equity'] > 0
    df.loc[mask, 'total_debt'] = df.loc[mask, 'total_equity'] * assume_dte

    # Actifs totaux
    if 'total_assets' not in df.columns:
        df['total_assets'] = np.nan
    mask = (df['total_equity'] > 0) | (df['total_debt'] > 0)
    df.loc[mask, 'total_assets'] = df.loc[mask, 'total_equity'].fillna(0) + df.loc[mask, 'total_debt'].fillna(0)

    # Ratios financiers avec validations
    df['Dividend_Yield_%'] = np.where(last_close > 0, 100 * df['dividend_per_share'] / last_close, 0)
    df['ROE_%'] = np.where(df['total_equity'] > 0, 100 * df['net_income'] / df['total_equity'], np.nan)
    df['Debt_to_Equity'] = np.where(df['total_equity'] > 0, df['total_debt'] / df['total_equity'], np.nan)
    df['Payout_%'] = np.where(df['net_income'] > 0, 100 * df['dividends_total'] / df['net_income'], np.nan)

    # Score fondamental corrigé
    def score_row(r):
        score = 0
        # EPS positif
        if pd.notna(r.get('EPS')) and r.get('EPS', 0) > 0:
            score += 2
        # PER raisonnable
        per = r.get('PER')
        if pd.notna(per):
            if 8 <= per <= 25:
                score += 2
            elif 5 <= per < 8 or 25 < per <= 40:
                score += 1
        # ROE élevé
        roe = r.get('ROE_%')
        if pd.notna(roe):
            if roe >= 20:
                score += 2
            elif roe >= 10:
                score += 1
        # Endettement maîtrisé
        dte = r.get('Debt_to_Equity')
        if pd.notna(dte):
            if dte <= 0.3:
                score += 2
            elif dte <= 0.8:
                score += 1
        # Rendement dividende
        dy = r.get('Dividend_Yield_%')
        if pd.notna(dy) and dy >= 2:
            score += 1
        return min(score, 10)

    df['Score_Fondamental_0_10'] = df.apply(score_row, axis=1)
    return df

def plot_fundamentals_summary(df_ratios: pd.DataFrame) -> go.Figure:
    """Graphique fondamentaux avec thème sombre"""
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=['Chiffre d\'affaires & Résultat Net', 'Price Earnings Ratio (PER)', 'Return on Equity (ROE %)', 'Score Fondamental'],
        vertical_spacing=0.25, horizontal_spacing=0.18,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # CA et Résultat Net
    if 'revenue' in df_ratios.columns:
        fig.add_trace(go.Scatter(
            x=df_ratios['period'], y=df_ratios['revenue'], 
            name='CA', mode='lines+markers',
            line=dict(color='#00d4ff', width=2), marker=dict(size=6)
        ), row=1, col=1)
    
    if 'net_income' in df_ratios.columns:
        fig.add_trace(go.Scatter(
            x=df_ratios['period'], y=df_ratios['net_income'], 
            name='Résultat Net', mode='lines+markers',
            line=dict(color='#ff6b35', width=2), marker=dict(size=6)
        ), row=1, col=1)
    
    # PER
    dfp = df_ratios[['period','PER']].replace([np.inf, -np.inf], np.nan).dropna()
    if not dfp.empty:
        fig.add_trace(go.Scatter(
            x=dfp['period'], y=dfp['PER'], name='PER', mode='lines+markers',
            line=dict(color='#00ff88', width=2), marker=dict(size=6)
        ), row=1, col=2)
        fig.add_hrect(y0=8, y1=25, fillcolor="green", opacity=0.1, line_width=0, row=1, col=2)
    
    # ROE
    if 'ROE_%' in df_ratios.columns:
        dfr = df_ratios[['period','ROE_%']].dropna()
        if not dfr.empty:
            fig.add_trace(go.Scatter(
                x=dfr['period'], y=dfr['ROE_%'], name='ROE', mode='lines+markers',
                line=dict(color='#ffff00', width=2), marker=dict(size=6)
            ), row=2, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.6, row=2, col=1)
            fig.add_hline(y=10, line_dash="dot", line_color="orange", opacity=0.6, row=2, col=1)
    
    # Score
    if 'Score_Fondamental_0_10' in df_ratios.columns:
        dfs = df_ratios[['period','Score_Fondamental_0_10']].dropna()
        if not dfs.empty:
            colors = ['#ff4444' if score < 4 else '#ffaa00' if score < 7 else '#00ff88' 
                     for score in dfs['Score_Fondamental_0_10']]
            fig.add_trace(go.Bar(
                x=dfs['period'], y=dfs['Score_Fondamental_0_10'], name='Score',
                marker_color=colors, opacity=0.8,
                text=dfs['Score_Fondamental_0_10'].round(1), textposition='auto'
            ), row=2, col=2)
    
    # Layout avec thème sombre
    fig.update_layout(
        height=480, showlegend=False, 
        margin=dict(t=100, b=80, l=80, r=80),
        plot_bgcolor='#2d3748', paper_bgcolor='#2d3748',
        font=dict(size=9, color='white'), title_font_size=11
    )
    
    # Titres en blanc
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11, color='white')
    
    # Grille sombre
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#4a5568')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#4a5568')
    
    return fig

def commentaire_auto_points(df_ratios: pd.DataFrame) -> List[str]:
    """Analyse automatique des ratios financiers"""
    notes = []
    if df_ratios.empty or 'period' not in df_ratios.columns:
        return ["Aucune donnée fondamentale disponible."]
    
    last = df_ratios.sort_values('period').iloc[-1]
    p = str(last.get('period'))

    eps = last.get('EPS', np.nan)
    if pd.notna(eps) and eps > 0:
        notes.append(f"**{p} — EPS positif** : {eps:,.2f} FCFA/action")
    elif pd.notna(eps):
        notes.append(f"**{p} — EPS négatif** : {eps:,.2f} FCFA/action - Performance préoccupante")

    per = last.get('PER', np.nan)
    if pd.notna(per):
        if 8 <= per <= 25:
            notes.append(f"**{p} — PER attractif** : {per:.1f}x - Valorisation équilibrée")
        elif per < 8:
            notes.append(f"**{p} — PER très bas** : {per:.1f}x - Possible opportunité ou signal d'alarme")
        else:
            notes.append(f"**{p} — PER élevé** : {per:.1f}x - Valorisation tendue")

    roe = last.get('ROE_%', np.nan)
    if pd.notna(roe):
        if roe >= 20:
            notes.append(f"**{p} — ROE excellent** : {roe:.1f}% - Très bonne rentabilité")
        elif roe >= 10:
            notes.append(f"**{p} — ROE correct** : {roe:.1f}% - Performance acceptable")
        else:
            notes.append(f"**{p} — ROE faible** : {roe:.1f}% - Rentabilité insuffisante")

    score = last.get('Score_Fondamental_0_10', np.nan)
    if pd.notna(score):
        if score >= 8:
            notes.append(f"**{p} — Score excellent** : {score:.1f}/10 - Fondamentaux très solides")
        elif score >= 6:
            notes.append(f"**{p} — Score bon** : {score:.1f}/10 - Fondamentaux corrects")
        elif score >= 4:
            notes.append(f"**{p} — Score moyen** : {score:.1f}/10 - Fondamentaux mitigés")
        else:
            notes.append(f"**{p} — Score faible** : {score:.1f}/10 - Fondamentaux préoccupants")

    if not notes:
        notes = [f"Données {p} présentes mais incomplètes pour l'analyse"]
    
    return notes

# --------------------------- APP PRINCIPALE ---------------------------
def main():
    st.title("Dashboard CFAOCI - BRVM")
    st.markdown("**Analyse technique et fondamentale de CFAO CI (Version Corrigée)**")
    
    # Avertissement sur les données
    with st.expander("⚠️ AVERTISSEMENT - Qualité des Données", expanded=False):
        st.error("""
        **ATTENTION :** Ce dashboard utilise des données de démonstration et des estimations.
        
        **Problèmes identifiés :**
        - Les prix peuvent ne pas correspondre aux cours réels BRVM
        - Les données fondamentales sont partiellement estimées avec des hypothèses
        - Les ratios PER et rendements peuvent contenir des erreurs
        
        **Recommandations :**
        - Vérifiez toujours avec les sources officielles BRVM avant toute décision
        - Consultez les rapports annuels officiels de CFAO CI
        - Ce tool est à usage éducatif et de démonstration uniquement
        """)
    
    # SIDEBAR
    with st.sidebar:
        st.header("Contrôles Dashboard")
        
        # Upload des données
        st.subheader("Source de Données")
        uploader = st.file_uploader("Importer CSV Prix", type=['csv'], key="price_csv")
        
        if uploader is not None:
            df_raw = load_data(uploader)
        else:
            # Données de test par défaut
            test_data = {
                'Date': pd.date_range('2024-01-01', periods=100, freq='D'),
                'Open': np.random.normal(660, 20, 100).clip(600, 720),
                'Close': np.random.normal(660, 20, 100).clip(600, 720),
                'Volume': np.random.normal(50000, 10000, 100).clip(10000, 200000)
            }
            df_raw = pd.DataFrame(test_data)
            df_raw['High'] = df_raw[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 10, len(df_raw))
            df_raw['Low'] = df_raw[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 10, len(df_raw))
            df_raw['Variation'] = df_raw['Close'].pct_change().fillna(0) * 100
            st.info("Utilisation de données de test aléatoires")
        
        if df_raw.empty:
            st.error("Impossible de charger les données")
            st.stop()
        
        # Contrôles temporels
        st.subheader("Période d'Analyse")
        
        # CORRECTION MAJEURE: Sélection de fréquence avec callback
        frequency = st.selectbox(
            "Fréquence d'affichage", 
            ['Jour', 'Semaine', 'Mois'], 
            key="frequency_selector"
        )
        
        # Sélection de dates
        min_date, max_date = df_raw['Date'].min().date(), df_raw['Date'].max().date()
        date_range = st.date_input(
            "Plage de dates", 
            value=(min_date, max_date),
            min_value=min_date, 
            max_value=max_date
        )
        
        # Traitement des dates
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        else:
            start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)
        
        # Filtrage et rééchantillonnage - CORRECTION MAJEURE
        df_filtered = df_raw[(df_raw['Date'] >= start_date) & (df_raw['Date'] <= end_date)]
        df_resampled = resample_data_by_frequency(df_filtered, frequency)
        
        # Information sur les données traitées
        st.success(f"""
        **Données chargées :**
        - Période: {df_resampled['Date'].min().strftime('%d/%m/%Y')} → {df_resampled['Date'].max().strftime('%d/%m/%Y')}
        - Fréquence: {frequency}
        - Points de données: {len(df_resampled)}
        """)
        
        # Indicateurs techniques
        st.subheader("Indicateurs Techniques")
        indicators = st.multiselect(
            "Sélection des indicateurs", 
            ['MM', 'EMA', 'Bollinger', 'RSI', 'MACD'], 
            default=['MM', 'RSI']
        )
        
        # Paramètres des indicateurs
        with st.expander("Paramètres Avancés", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                sma1 = st.slider("MM Courte", 5, 50, 20)
                sma2 = st.slider("MM Longue", 20, 200, 50)
                ema1 = st.slider("EMA", 5, 50, 20)
                bb_window = st.slider("Bollinger Période", 10, 50, 20)
            with col2:
                bb_std = st.slider("Bollinger Écart-Type", 1.0, 3.0, 2.0, 0.1)
                rsi_window = st.slider("RSI Période", 5, 30, 14)
                macd_fast = st.slider("MACD Rapide", 5, 20, 12)
                macd_slow = st.slider("MACD Lent", 15, 40, 26)
        
        params = {
            'show_sma': 'MM' in indicators, 'sma1': sma1, 'sma2': sma2,
            'show_ema': 'EMA' in indicators, 'ema1': ema1,
            'show_bb': 'Bollinger' in indicators, 'bb_window': bb_window, 'bb_std': bb_std,
            'show_rsi': 'RSI' in indicators, 'rsi_window': rsi_window,
            'show_macd': 'MACD' in indicators, 'macd_fast': macd_fast, 'macd_slow': macd_slow, 'macd_signal': 9
        }
        
        # Style de graphique
        st.subheader("Affichage")
        chart_type = st.radio("Type de graphique", ['Ligne', 'Chandelles'], index=1)
        rf_rate = st.number_input("Taux sans risque (%)", value=2.5, step=0.1, format="%.1f")
        
        # Backtest
        st.subheader("Simulation de Trading")
        enable_backtest = st.checkbox("Activer le backtest", value=False)
        if enable_backtest:
            backtest_strategy = st.selectbox("Stratégie", ["sma_crossover"], index=0)
        
        # Fondamentaux
        st.subheader("Données Fondamentales")
        fund_uploader = st.file_uploader("CSV Fondamentaux", type=['csv'], key="fund_csv")
        
        col1, col2 = st.columns(2)
        with col1:
            assume_roe = st.slider("ROE Estimé (%)", 5, 30, 15) / 100
        with col2:
            assume_dte = st.slider("Ratio D/E", 0.0, 2.0, 0.4, 0.1)

    # TRAITEMENT DES DONNÉES
    df_with_indicators = add_indicators(df_resampled, params)
    metrics = performance_metrics(df_with_indicators, rf_rate)
    
    # Données fondamentales
    if fund_uploader is not None:
        try:
            df_fund = load_fundamentals(fund_uploader)
        except:
            df_fund = fundamentals_default_df()
    else:
        df_fund = fundamentals_default_df()
    
    current_price = metrics.get('current_price', 660.0)
    df_ratios = impute_fundamentals(df_fund, assume_roe, assume_dte, current_price)

    # AFFICHAGE PRINCIPAL
    
    # Métriques de performance
    st.subheader("Métriques de Performance")
    if metrics:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Prix Actuel", f"{current_price:.0f} FCFA")
        col2.metric("Rendement Total", f"{metrics.get('total_return', 0):.1f}%")
        col3.metric("Rendement Annuel", f"{metrics.get('annualized_return', 0):.1f}%")
        col4.metric("Volatilité", f"{metrics.get('volatility', 0):.1f}%")
        col5.metric("Drawdown Max", f"{metrics.get('max_drawdown', 0):.1f}%")
        col6.metric("Ratio Sharpe", f"{metrics.get('sharpe', 0):.2f}")
    
    # Layout principal en colonnes
    col_chart, col_fundamentals = st.columns([3, 2])
    
    with col_chart:
        st.subheader(f"Graphique de Prix - {frequency}")
        chart_fig = create_advanced_candlestick_chart(df_with_indicators, chart_type, params)
        # Appliquer le thème sombre au graphique principal aussi
        chart_fig.update_layout(
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font_color='white'
        )
        chart_fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='#404040',
            title_font_color='white', tickfont_color='white'
        )
        chart_fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor='#404040',
            title_font_color='white', tickfont_color='white'
        )
        st.plotly_chart(chart_fig, use_container_width=True, config={"displaylogo": False})
    
    with col_fundamentals:
        st.subheader("Analyse Fondamentale")
        fund_fig = plot_fundamentals_summary(df_ratios)
        st.plotly_chart(fund_fig, use_container_width=True, config={"displaylogo": False})
        
        # Commentaires automatiques
        st.markdown("**Synthèse Fondamentale :**")
        for comment in commentaire_auto_points(df_ratios):
            st.write(f"• {comment}")
    
    # Section analyse et backtest
    col_analysis, col_backtest = st.columns([1, 1])
    
    with col_analysis:
        st.subheader("Analyse Technique")
        if not df_with_indicators.empty:
            latest = df_with_indicators.iloc[-1]
            signals = []
            
            if 'RSI' in df_with_indicators.columns:
                rsi = float(latest['RSI'])
                if rsi > 70:
                    signals.append(f"**RSI Surachat** ({rsi:.1f}) - Signal de vente potentiel")
                elif rsi < 30:
                    signals.append(f"**RSI Survente** ({rsi:.1f}) - Signal d'achat potentiel")
                else:
                    signals.append(f"**RSI Neutre** ({rsi:.1f}) - Pas de signal clair")
            
            if 'SMA_1' in df_with_indicators.columns and 'SMA_2' in df_with_indicators.columns:
                if latest['Close'] > latest['SMA_1'] > latest['SMA_2']:
                    signals.append("**Tendance Haussière** - Prix > MM20 > MM50")
                elif latest['Close'] < latest['SMA_1'] < latest['SMA_2']:
                    signals.append("**Tendance Baissière** - Prix < MM20 < MM50")
                else:
                    signals.append("**Tendance Neutre** - Moyennes mobiles entremêlées")
            
            for signal in signals:
                st.info(signal)
    
    with col_backtest:
        st.subheader("Résultats Backtest")
        if enable_backtest and len(df_with_indicators) >= 100:
            backtest_results = advanced_backtest_strategy(df_with_indicators, backtest_strategy)
            
            if "error" not in backtest_results:
                st.success("**Performance du Backtest :**")
                
                col_bt1, col_bt2 = st.columns(2)
                with col_bt1:
                    st.metric("Rendement Total", f"{backtest_results['total_return']:.2f}%")
                    st.metric("Nb Trades", backtest_results['total_trades'])
                with col_bt2:
                    st.metric("Taux Réussite", f"{backtest_results['win_rate']:.1f}%")
                    st.metric("Drawdown Max", f"{backtest_results['max_drawdown']:.2f}%")
                
                if backtest_results['total_trades'] > 0:
                    st.metric("Ratio Sharpe", f"{backtest_results['sharpe_ratio']:.2f}")
                    
                    # Affichage des derniers trades
                    if backtest_results["trades"]:
                        st.write("**Derniers Trades :**")
                        recent_trades = backtest_results["trades"][-3:]
                        for trade in recent_trades:
                            profit_color = "🟢" if trade["pnl_pct"] > 0 else "🔴"
                            st.write(f"{profit_color} {trade['entry_date'].strftime('%d/%m')} → {trade['exit_date'].strftime('%d/%m')}: {trade['pnl_pct']:.1f}%")
            else:
                st.error(f"Erreur backtest: {backtest_results['error']}")
        elif enable_backtest:
            st.warning("Données insuffisantes pour le backtest (minimum 100 points)")
    
    # Section téléchargements
    st.subheader("Téléchargements")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        # CSV des prix
        csv_data = df_with_indicators[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        csv_data['Date'] = csv_data['Date'].dt.strftime('%Y-%m-%d')
        csv_prices = csv_data.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger Prix CSV", csv_prices, f"CFAOCI_prix_{frequency.lower()}.csv", "text/csv")
    
    with col_dl2:
        # CSV fondamentaux
        fund_cols = ['period', 'revenue', 'net_income', 'EPS', 'PER', 'ROE_%', 'Score_Fondamental_0_10']
        csv_fund = df_ratios[[c for c in fund_cols if c in df_ratios.columns]].to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger Fondamentaux CSV", csv_fund, "CFAOCI_fondamentaux.csv", "text/csv")
    
    with col_dl3:
        # Rapport d'analyse
        report = f"""# Rapport d'Analyse CFAOCI - {datetime.now().strftime('%d/%m/%Y')}

## Données Techniques
- Période: {frequency}
- Prix actuel: {current_price:.0f} FCFA
- Rendement total: {metrics.get('total_return', 0):.2f}%
- Volatilité: {metrics.get('volatility', 0):.1f}%

## Analyse Fondamentale
{chr(10).join([f"- {comment.replace('**', '')}" for comment in commentaire_auto_points(df_ratios)])}

## Avertissement
Cette analyse est basée sur des données de démonstration et ne constitue pas un conseil d'investissement.
Consultez toujours les sources officielles BRVM avant toute décision financière.
"""
        st.download_button("Télécharger Rapport", report.encode('utf-8'), "rapport_CFAOCI.md", "text/markdown")
    
    # Pied de page avec informations importantes
    st.markdown("---")
    st.markdown("""
    <div style='background-color: #f0f0f0; padding: 15px; border-radius: 8px; margin-top: 20px;'>
        <h4 style='color: #d63384; margin-bottom: 10px;'>⚠️ CORRECTIONS APPORTÉES DANS CETTE VERSION</h4>
        <ul style='color: #495057; font-size: 14px;'>
            <li><strong>Fréquences:</strong> Rééchantillonnage correct pour semaine/mois - les graphiques se mettent à jour</li>
            <li><strong>Chandelles:</strong> Affichage correct des mèches (High/Low) avec validation OHLC</li>
            <li><strong>Backtest:</strong> Système de simulation avec stratégie croisement MM20/50 et métriques détaillées</li>
            <li><strong>Données:</strong> Validation et correction des formules PER, ratios financiers et prix OHLC</li>
            <li><strong>Interface:</strong> Messages d'alerte sur la qualité des données et sources recommandées</li>
        </ul>
        <p style='margin-top: 10px; font-style: italic; color: #6c757d;'>
            <strong>Dashboard CFAOCI - BRVM</strong> | Version Corrigée avec Validation des Données
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()# memoire_gueri_dashboard_interactif.py
        ["2020",  99126, 3780, 181_371_900, np.nan, 22.15, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2021", 119731, 6711

