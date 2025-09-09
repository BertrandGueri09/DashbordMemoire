# memoire_gueri_dashboard_interactif.py
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

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# --------------------------- UTILITAIRES TECHNIQUES ---------------------------
@st.cache_data
def load_data(path_or_buffer: str | io.BytesIO) -> pd.DataFrame:
    """Charger et traiter les données CSV (prix)"""
    df = pd.read_csv(path_or_buffer)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

    rename_map = {'Dernier': 'Close', 'Ouv.': 'Open', 'Plus Haut': 'High', 'Plus Bas': 'Low'}
    df = df.rename(columns=rename_map)

    for col in ['Close', 'Open', 'High', 'Low']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    def parse_volume(vol_str):
        if pd.isna(vol_str) or vol_str == '':
            return 0.0
        s = str(vol_str).replace(',', '.').strip()
        try:
            if s.endswith('K'):
                return float(s[:-1]) * 1_000
            if s.endswith('M'):
                return float(s[:-1]) * 1_000_000
            return float(s)
        except:
            return 0.0

    if 'Vol.' in df.columns:
        df['Volume'] = df['Vol.'].apply(parse_volume)
    elif 'Volume' not in df.columns:
        df['Volume'] = 0.0

    def parse_variation(var_str):
        if pd.isna(var_str) or var_str == '':
            return 0.0
        try:
            return float(str(var_str).replace('%', '').replace(',', '.'))
        except:
            return 0.0

    if 'Variation %' in df.columns:
        df['Variation'] = df['Variation %'].apply(parse_variation)
    elif 'Variation' not in df.columns:
        df['Variation'] = 0.0

    df = df.dropna(subset=['Date', 'Close', 'Open', 'High', 'Low'])
    df = df.sort_values('Date').reset_index(drop=True)
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
    latest = df.iloc[-1]
    oldest = df.iloc[0]
    total_return = ((latest['Close'] - oldest['Close']) / oldest['Close']) * 100
    n = len(df)
    ann_return = ((latest['Close'] / oldest['Close']) ** (252 / max(n, 1)) - 1) * 100

    daily_ret = df['Close'].pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252) * 100
    rf_daily = (rf_annual_pct / 100) / 252
    sharpe = 0.0
    if daily_ret.std() > 0:
        sharpe = ((daily_ret.mean() - rf_daily) / daily_ret.std()) * np.sqrt(252)

    cummax = df['Close'].cummax()
    dd = df['Close'] / cummax - 1.0
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

def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    dfi = df.set_index('Date')
    agg = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Variation': 'mean'
    }
    out = dfi.resample(freq).agg(agg).dropna().reset_index()
    return out

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
        df['RSI'] = calculate_rsi(df['Close'], params['rsi_window'])
    if params.get('show_macd'):
        macd_l, macd_s, macd_h = macd(df['Close'], params['macd_fast'], params['macd_slow'], params['macd_signal'])
        df['MACD_L'], df['MACD_S'], df['MACD_H'] = macd_l, macd_s, macd_h
    return df

def plotly_combined_chart(df: pd.DataFrame, chart_type: str, params: Dict) -> go.Figure:
    """Graphique combiné avec indicateurs dans des sous-graphiques"""
    # Déterminer le nombre de lignes nécessaires
    rows = 1
    if params.get('show_rsi'): rows += 1
    if params.get('show_macd'): rows += 1
    
    # Hauteurs relatives
    if rows == 1:
        row_heights = [1.0]
    elif rows == 2:
        row_heights = [0.7, 0.3]
    else:
        row_heights = [0.6, 0.2, 0.2]
    
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=['Prix & Volume'] + 
                      (['RSI'] if params.get('show_rsi') else []) + 
                      (['MACD'] if params.get('show_macd') else [])
    )
    
    current_row = 1
    
    # Prix principal
    if chart_type == 'Chandelles':
        fig.add_trace(
            go.Candlestick(
                x=df['Date'], open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Cours'
            ), row=current_row, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Prix', line=dict(width=2)),
            row=current_row, col=1
        )
    
    # Moyennes mobiles
    if params.get('show_sma'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_1'], name=f"MM{params['sma1']}", mode='lines'), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_2'], name=f"MM{params['sma2']}", mode='lines'), row=current_row, col=1)
    if params.get('show_ema'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_1'], name=f"EMA{params['ema1']}", mode='lines'), row=current_row, col=1)
    
    # Bandes de Bollinger
    if params.get('show_bb'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_M'], name="BB", mode='lines', line=dict(dash='dot')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_U'], mode='lines', line=dict(width=0), showlegend=False), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_L'], fill='tonexty', mode='lines', line=dict(width=0), name='BB Zone', opacity=0.1), row=current_row, col=1)
    
    current_row += 1
    
    # RSI
    if params.get('show_rsi'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', mode='lines'), row=current_row, col=1)
        fig.add_hline(y=70, line_dash='dash', line_color='red', opacity=0.6, row=current_row, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='green', opacity=0.6, row=current_row, col=1)
        fig.add_hline(y=50, line_dash='dot', line_color='gray', opacity=0.4, row=current_row, col=1)
        current_row += 1
    
    # MACD
    if params.get('show_macd'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_L'], name='MACD', mode='lines'), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_S'], name='Signal', mode='lines'), row=current_row, col=1)
        fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_H'], name='Hist', opacity=0.6), row=current_row, col=1)
    
    fig.update_layout(
        height=550, hovermode='x unified', showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(t=40, b=40, l=40, r=40)
    )
    return fig

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
    """Jeu de données fondamentales intégré par défaut (2020 → 2025)."""
    data = [
        ["2020",  99126, 3780, 181_371_900, np.nan, 22.15, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2021", 119731, 6711, 181_371_900, np.nan, 69.47, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2022", 146375, 5534, 181_371_900, np.nan, 28.67, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2023", 180162, 6399, 181_371_900, np.nan, 15.88, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2024", 158313, 4693, 181_371_900, np.nan,  7.04, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2025",     np.nan,   np.nan, 181_371_900, np.nan,   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]
    cols = ["period","revenue","net_income","shares_outstanding","dividends_total","dividend_per_share","total_equity","total_debt","total_assets","cash_and_equivalents","capex","EPS"]
    return pd.DataFrame(data, columns=cols)

def _fit_yearly_trend_impute(df: pd.DataFrame, col: str) -> pd.Series:
    """Impute la colonne 'col' via régression linéaire simple (année -> valeur) si possible, sinon ffill/bfill."""
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
            out = out.copy()
            out = out.where(out.notna(), pred)
    except Exception:
        pass
    out = out.ffill().bfill()
    return out

def impute_fundamentals(df_fund: pd.DataFrame, assume_roe: float, assume_dte: float, last_close: float) -> pd.DataFrame:
    """Complète les NaN avec des hypothèses raisonnables"""
    df = df_fund.copy()
    df['period'] = df['period'].astype(str)

    if 'revenue' in df.columns:
        df['revenue'] = _fit_yearly_trend_impute(df, 'revenue')
    if 'net_income' in df.columns:
        df['net_income'] = _fit_yearly_trend_impute(df, 'net_income')

    if 'dividend_per_share' in df.columns:
        df['dividend_per_share'] = df['dividend_per_share'].ffill().bfill()

    if 'EPS' not in df.columns:
        df['EPS'] = np.nan
    if {'net_income', 'shares_outstanding'} <= set(df.columns):
        df['EPS'] = df['EPS'].where(df['EPS'].notna(), df['net_income'] / df['shares_outstanding'])

    df['PER'] = last_close / df['EPS'].replace(0, np.nan)

    if {'dividend_per_share', 'shares_outstanding'} <= set(df.columns):
        if 'dividends_total' not in df.columns:
            df['dividends_total'] = np.nan
        df['dividends_total'] = df['dividends_total'].where(df['dividends_total'].notna(), df['dividend_per_share'] * df['shares_outstanding'])

    if 'total_equity' not in df.columns:
        df['total_equity'] = np.nan
    if 'net_income' in df.columns:
        roe = max(assume_roe, 1e-6)
        df['total_equity'] = df['total_equity'].where(df['total_equity'].notna(), df['net_income'] / roe)

    if 'total_debt' not in df.columns:
        df['total_debt'] = np.nan
    df['total_debt'] = df['total_debt'].where(df['total_debt'].notna(), assume_dte * df['total_equity'])

    if 'total_assets' not in df.columns:
        df['total_assets'] = np.nan
    df['total_assets'] = df['total_assets'].where(df['total_assets'].notna(), df['total_equity'] + df['total_debt'])

    for c in ['cash_and_equivalents', 'capex']:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0.0)

    df['Dividend_Yield_%'] = 100 * df['dividend_per_share'] / last_close
    df['ROE_%'] = 100 * df['net_income'] / df['total_equity'].replace(0, np.nan)
    df['Debt_to_Equity'] = df['total_debt'] / df['total_equity'].replace(0, np.nan)
    df['Payout_%'] = 100 * df['dividends_total'] / df['net_income'].replace(0, np.nan)

    def score_row(r):
        score = 0
        if pd.notna(r.get('EPS')) and r.get('EPS', 0) > 0:
            score += 1
        per = r.get('PER')
        if pd.notna(per):
            if 5 <= per <= 20:
                score += 2
            elif per < 5:
                score += 1
        roe = r.get('ROE_%')
        if pd.notna(roe):
            if roe >= 15:
                score += 2
            elif roe >= 8:
                score += 1
        dte = r.get('Debt_to_Equity')
        if pd.notna(dte):
            if dte <= 0.5:
                score += 2
            elif dte <= 1:
                score += 1
        dy = r.get('Dividend_Yield_%')
        if pd.notna(dy):
            if dy >= 4:
                score += 2
            elif dy >= 2:
                score += 1
        return min(score, 10)

    df['Score_Fondamental_0_10'] = df.apply(score_row, axis=1)
    return df

def plot_fundamentals_summary(df_ratios: pd.DataFrame) -> go.Figure:
    """Graphique résumé des fondamentaux avec espacement amélioré"""
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=['Chiffre d\'affaires & Résultat Net', 'Price Earnings Ratio (PER)', 'Return on Equity (ROE %)', 'Score Fondamental'],
        vertical_spacing=0.15, horizontal_spacing=0.12,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # CA et Résultat Net avec couleurs distinctes
    if 'revenue' in df_ratios.columns:
        fig.add_trace(go.Scatter(
            x=df_ratios['period'], 
            y=df_ratios['revenue'], 
            name='Chiffre d\'affaires', 
            mode='lines+markers',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ), row=1, col=1)
    
    if 'net_income' in df_ratios.columns:
        fig.add_trace(go.Scatter(
            x=df_ratios['period'], 
            y=df_ratios['net_income'], 
            name='Résultat Net', 
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=8)
        ), row=1, col=1)
    
    # PER avec zone de valorisation raisonnable
    dfp = df_ratios[['period','PER']].replace([np.inf, -np.inf], np.nan).dropna()
    if not dfp.empty:
        fig.add_trace(go.Scatter(
            x=dfp['period'], 
            y=dfp['PER'], 
            name='PER', 
            mode='lines+markers',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=8)
        ), row=1, col=2)
        
        # Zone de valorisation raisonnable (PER entre 10-20)
        fig.add_hrect(y0=10, y1=20, 
                      fillcolor="lightgreen", opacity=0.1, 
                      line_width=0, row=1, col=2)
        
        # Annotations pour les zones
        fig.add_annotation(x=dfp['period'].iloc[-1], y=15, 
                          text="Zone raisonnable", showarrow=False, 
                          font=dict(size=10, color="green"), row=1, col=2)
    
    # ROE avec seuils de performance
    if 'ROE_%' in df_ratios.columns:
        dfr = df_ratios[['period','ROE_%']].dropna()
        if not dfr.empty:
            fig.add_trace(go.Scatter(
                x=dfr['period'], 
                y=dfr['ROE_%'], 
                name='ROE (%)', 
                mode='lines+markers',
                line=dict(color='#d62728', width=3),
                marker=dict(size=8)
            ), row=2, col=1)
            
            # Seuil de performance élevée (ROE > 15%)
            fig.add_hline(y=15, line_dash="dash", line_color="green", 
                         opacity=0.6, row=2, col=1)
            fig.add_hline(y=8, line_dash="dot", line_color="orange", 
                         opacity=0.6, row=2, col=1)
            
            # Annotations
            fig.add_annotation(x=dfr['period'].iloc[-1], y=15, 
                              text="ROE élevé", showarrow=False, 
                              font=dict(size=10, color="green"), 
                              xshift=20, row=2, col=1)
    
    # Score avec code couleur
    if 'Score_Fondamental_0_10' in df_ratios.columns:
        dfs = df_ratios[['period','Score_Fondamental_0_10']].dropna()
        if not dfs.empty:
            # Couleurs basées sur le score
            colors = []
            for score in dfs['Score_Fondamental_0_10']:
                if score >= 8:
                    colors.append('#2ecc71')  # Vert pour excellent
                elif score >= 6:
                    colors.append('#f39c12')  # Orange pour bon
                elif score >= 4:
                    colors.append('#e74c3c')  # Rouge pour moyen
                else:
                    colors.append('#95a5a6')  # Gris pour faible
            
            fig.add_trace(go.Bar(
                x=dfs['period'], 
                y=dfs['Score_Fondamental_0_10'], 
                name='Score',
                marker_color=colors,
                opacity=0.8,
                text=dfs['Score_Fondamental_0_10'].round(1),
                textposition='auto'
            ), row=2, col=2)
            
            # Ligne de référence pour score moyen
            fig.add_hline(y=5, line_dash="dash", line_color="gray", 
                         opacity=0.5, row=2, col=2)
    
    # Mise à jour des axes avec titres plus clairs
    fig.update_xaxes(title_text="Période", row=1, col=1)
    fig.update_yaxes(title_text="Montants (FCFA)", row=1, col=1)
    
    fig.update_xaxes(title_text="Période", row=1, col=2)
    fig.update_yaxes(title_text="PER (x)", row=1, col=2)
    
    fig.update_xaxes(title_text="Période", row=2, col=1)
    fig.update_yaxes(title_text="ROE (%)", row=2, col=1)
    
    fig.update_xaxes(title_text="Période", row=2, col=2)
    fig.update_yaxes(title_text="Score (0-10)", range=[0, 10], row=2, col=2)
    
    # Layout général amélioré
    fig.update_layout(
        height=420, 
        showlegend=False, 
        margin=dict(t=80, b=60, l=60, r=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=11),
        title_font_size=14
    )
    
    # Grille légère pour tous les sous-graphiques
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    
    return fig

def commentaire_auto_points(df_ratios: pd.DataFrame) -> List[str]:
    """Messages courts et clairs sur la dernière période renseignée."""
    notes = []
    if df_ratios.empty or 'period' not in df_ratios.columns:
        return ["Aucune donnée fondamentale disponible."]
    last = df_ratios.sort_values('period').iloc[-1]
    p = str(last.get('period'))

    eps = last.get('EPS', np.nan)
    if pd.notna(eps) and eps > 0:
        notes.append(f"**{p} — EPS positif** : {eps:,.2f} FCFA/action")
    per = last.get('PER', np.nan)
    if pd.notna(per):
        if 5 <= per <= 20:
            notes.append(f"**{p} — PER** ≈ {per:.1f} (raisonnable)")
        elif per < 5:
            notes.append(f"**{p} — PER** ≈ {per:.1f} (décote potentielle)")
        else:
            notes.append(f"**{p} — PER** ≈ {per:.1f} (valorisation tendue)")
    roe = last.get('ROE_%', np.nan)
    if pd.notna(roe):
        if roe >= 15:
            notes.append(f"**{p} — ROE élevé** : {roe:.1f}%")
        elif roe >= 8:
            notes.append(f"**{p} — ROE correct** : {roe:.1f}%")
        else:
            notes.append(f"**{p} — ROE faible** : {roe:.1f}%")
    score = last.get('Score_Fondamental_0_10', np.nan)
    if pd.notna(score):
        notes.append(f"**{p} — Score fondamental** : **{score:.1f}/10**")

    if not notes:
        notes = [f"Données {p} présentes mais incomplètes"]
    return notes

# --------------------------- APP ---------------------------
def main():
    st.title("Dashboard CFAOCI - BRVM")
    st.markdown("**Analyse technique et fondamentale de CFAO CI**")
    
    # SIDEBAR CONDENSÉ
    with st.sidebar:
        st.header("⚙Contrôles")
        
        # Données
        uploader = st.file_uploader("CSV Prix (opt.)", type=['csv'], key="price_csv")
        if uploader is not None:
            df = load_data(uploader)
        else:
            try:
                df = load_data('CFAOCI.csv')
            except Exception:
                st.error("❌ Impossible de charger les données")
                st.stop()
        
        # Période
        st.subheader("Période")
        freq = st.selectbox("Fréquence", ['Jour', 'Semaine', 'Mois'])
        freq_map = {'Jour': 'D', 'Semaine': 'W', 'Mois': 'M'}
        
        min_date, max_date = df['Date'].min().date(), df['Date'].max().date()
        date_range = st.date_input("Dates", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        
        if isinstance(date_range, tuple):
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        else:
            start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)
        
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        df = resample_ohlcv(df, freq_map[freq])
        
        # Indicateurs
        st.subheader("Indicateurs")
        indicators = st.multiselect("Sélection", ['MM', 'EMA', 'Bollinger', 'RSI', 'MACD'], default=['MM', 'RSI'])
        
        # Paramètres compacts
        with st.expander("Paramètres", expanded=False):
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
        
        # Style
        st.subheader("Style")
        chart_type = st.radio("Type", ['Ligne', 'Chandelles'])
        rf = st.number_input("Taux sans risque (%)", value=2.0, step=0.5)
        
        # Fondamentaux
        st.subheader("Fondamentaux")
        fund_uploader = st.file_uploader("CSV Fond. (opt.)", type=['csv'], key="fund_csv")
        col1, col2 = st.columns(2)
        with col1:
            assume_roe_pct = st.slider("ROE (%)", 5, 25, 12, 1)
        with col2:
            assume_dte = st.slider("D/E", 0.0, 2.0, 0.60, 0.05)

    # TRAITEMENT DES DONNÉES
    df = add_indicators(df, params)
    metrics = performance_metrics(df, rf_annual_pct=rf)
    
    if fund_uploader is not None:
        try:
            df_fund = load_fundamentals(fund_uploader)
        except Exception:
            df_fund = fundamentals_default_df()
    else:
        df_fund = fundamentals_default_df()
    
    last_close = float(metrics['current_price'])
    df_ratios = impute_fundamentals(df_fund, assume_roe=assume_roe_pct/100.0, assume_dte=assume_dte, last_close=last_close)

    # MÉTRIQUES
    st.subheader("Métriques Principales")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Prix Actuel", f"{metrics['current_price']:.0f} FCFA")
    col2.metric("Rendement Total", f"{metrics['total_return']:.1f}%")
    col3.metric("Rend. Annualisé", f"{metrics['annualized_return']:.1f}%")
    col4.metric("Volatilité", f"{metrics['volatility']:.1f}%")
    col5.metric("Max DD", f"{metrics['max_drawdown']:.1f}%")
    col6.metric("Sharpe", f"{metrics['sharpe']:.2f}")

    # LAYOUT PRINCIPAL EN COLONNES
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("Graphique Principal")
        main_fig = plotly_combined_chart(df, chart_type, params)
        st.plotly_chart(main_fig, use_container_width=True, config={"displaylogo": False})
    
    with col_right:
        st.subheader("Analyse Fondamentale")
        fund_fig = plot_fundamentals_summary(df_ratios)
        st.plotly_chart(fund_fig, use_container_width=True, config={"displaylogo": False})
        
        # Commentaires auto
        st.markdown("**Résumé :**")
        for note in commentaire_auto_points(df_ratios):
            st.write(f"• {note}")

    # ANALYSE TECHNIQUE AUTO + TÉLÉCHARGEMENTS
    col_analysis, col_downloads = st.columns([2, 1])
    
    with col_analysis:
        st.subheader("🔍 Analyse Technique Auto")
        latest = df.iloc[-1]
        notes = []
        
        if 'RSI' in df.columns:
            rsi_value = float(latest['RSI'])
            if rsi_value > 70:
                notes.append(f"**RSI:** Surachat ({rsi_value:.1f})")
            elif rsi_value < 30:
                notes.append(f"**RSI:** Survente ({rsi_value:.1f})")
            else:
                notes.append(f"**RSI:** Neutre ({rsi_value:.1f})")
        
        if 'SMA_1' in df.columns and 'SMA_2' in df.columns:
            cond_up = latest['Close'] > latest['SMA_1'] > latest['SMA_2']
            cond_down = latest['Close'] < latest['SMA_1'] < latest['SMA_2']
            if cond_up:
                notes.append("**Tendance:** Haussière forte")
            elif cond_down:
                notes.append("**Tendance:** Baissière forte")
            else:
                notes.append("**Tendance:** Neutre/Consolidation")
        
        for note in notes:
            st.write(f"• {note}")
        
        st.info(f"**Min/Max période:** {df['Close'].min():.0f} - {df['Close'].max():.0f} FCFA | **Sessions:** {len(df)} | **Dernière MAJ:** {metrics['last_update']}")
    
    with col_downloads:
        st.subheader("Téléchargements")
        
        # CSV Prix filtrés
        display_df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Variation']].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%d/%m/%Y')
        csv_prix = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("CSV Prix", csv_prix, "CFAOCI_prix.csv", "text/csv")
        
        # CSV Fondamentaux
        cols_fund = [c for c in ['period', 'revenue', 'net_income', 'EPS', 'PER', 'ROE_%', 'Debt_to_Equity', 'Dividend_Yield_%', 'Score_Fondamental_0_10'] if c in df_ratios.columns]
        csv_fund = df_ratios[cols_fund].to_csv(index=False).encode('utf-8')
        st.download_button("CSV Fondamentaux", csv_fund, "CFAOCI_fondamentaux.csv", "text/csv")
        
        # Résumé Markdown
        def resume_markdown(df_ratios: pd.DataFrame) -> str:
            lines = ["# Synthèse CFAO CI", ""]
            if df_ratios.empty:
                return "Aucune donnée"
            last = df_ratios.sort_values('period').iloc[-1]
            p = str(last.get('period'))
            lines += [f"## Période {p}", ""]
            for msg in commentaire_auto_points(df_ratios):
                lines += [f"- {msg.replace('**', '')}"]
            lines += ["", "> *Analyse indicative*"]
            return "\n".join(lines)
        
        md_text = resume_markdown(df_ratios)
        st.download_button("Résumé MD", md_text.encode('utf-8'), "CFAOCI_resume.md", "text/markdown")

    # PIED DE PAGE
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
            <strong>Dashboard CFAOCI - BRVM</strong> | Analyse technique & fondamentale
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
