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
    page_title="Dashboard CFAOCI Trading  stock market - BRVM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

st.title("📈 Dashboard CFAOCI Trading  stock market - BRVM")
st.markdown("**Analyse technique et fondamentale de CFAO CI (BRVM)**")
st.markdown("---")

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

def plotly_main_chart(df: pd.DataFrame, chart_type: str, params: Dict) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )
    up = df['Close'] >= df['Open']
    colors = np.where(up, 'rgb(34,197,94)', 'rgb(239,68,68)')

    if chart_type == 'Chandelles japonaises':
        fig.add_trace(
            go.Candlestick(
                x=df['Date'], open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Cours'
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Prix de clôture'),
            row=1, col=1
        )

    if params.get('show_sma'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_1'], name=f"MM {params['sma1']}", mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_2'], name=f"MM {params['sma2']}", mode='lines'), row=1, col=1)
    if params.get('show_ema'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_1'], name=f"EMA {params['ema1']}", mode='lines'), row=1, col=1)
    if params.get('show_bb'):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_M'], name="BB moyenne", mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_U'], name="BB sup", mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_L'], name="BB inf", mode='lines'), row=1, col=1)
        fig.add_traces([
            go.Scatter(x=pd.concat([df['Date'], df['Date'][::-1]]),
                       y=pd.concat([df['BB_U'], df['BB_L'][::-1]]),
                       fill='toself', mode='none', name='Zone Bollinger', opacity=0.1)
        ], rows=[1], cols=[1])

    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color=colors, opacity=0.7), row=2, col=1)

    fig.update_layout(
        height=720,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        xaxis=dict(rangeslider=dict(visible=True), type='date')
    )
    fig.update_yaxes(title_text="Prix (FCFA)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

def plotly_rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', mode='lines'))
    for level, name, dash in [(70, 'Surachat (70)', 'dash'), (50, 'Neutre (50)', 'dot'), (30, 'Survente (30)', 'dash')]:
        fig.add_hline(y=level, line_dash=dash, annotation_text=name, annotation_position='top left', opacity=0.6)
    fig.update_layout(height=300, hovermode='x unified', yaxis=dict(range=[0, 100]), xaxis=dict(type='date'))
    return fig

def plotly_macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.4])
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_L'], name='MACD', mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_S'], name='Signal', mode='lines'), row=1, col=1)
    fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_H'], name='Histogramme', opacity=0.7), row=2, col=1)
    fig.update_layout(height=350, hovermode='x unified', xaxis=dict(type='date'))
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
        # period, revenue, net_income, shares_outstanding, dividends_total, dividend_per_share,
        # total_equity, total_debt, total_assets, cash_and_equivalents, capex, EPS
        ["2020",  99126, 3780, 181_371_900, np.nan, 22.15, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2021", 119731, 6711, 181_371_900, np.nan, 69.47, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2022", 146375, 5534, 181_371_900, np.nan, 28.67, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2023", 180162, 6399, 181_371_900, np.nan, 15.88, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2024", 158313, 4693, 181_371_900, np.nan,  7.04, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ["2025",     np.nan,   np.nan, 181_371_900, np.nan,   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]
    cols = [
        "period","revenue","net_income","shares_outstanding","dividends_total",
        "dividend_per_share","total_equity","total_debt","total_assets",
        "cash_and_equivalents","capex","EPS"
    ]
    return pd.DataFrame(data, columns=cols)

def _fit_yearly_trend_impute(df: pd.DataFrame, col: str) -> pd.Series:
    """Impute la colonne 'col' via régression linéaire simple (année -> valeur) si possible, sinon ffill/bfill."""
    s = df[['period', col]].dropna()
    out = df[col].copy()
    try:
        years = pd.to_numeric(s['period'], errors='coerce')
        mask = years.notna() & s[col].notna()
        if mask.sum() >= 2:
            # Régression linéaire (moindre carrés)
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
    # Si encore des NaN : ffill/bfill
    out = out.ffill().bfill()
    return out

def impute_fundamentals(df_fund: pd.DataFrame, assume_roe: float, assume_dte: float, last_close: float) -> pd.DataFrame:
    """
    Complète les NaN avec des hypothèses raisonnables :
    - revenue / net_income : régression sur l'historique puis ffill/bfill
    - dividend_per_share : ffill (si NaN)
    - dividends_total : = DPS * shares_outstanding si manquant
    - EPS : = net_income / shares_outstanding
    - total_equity : = net_income / assume_roe si manquant
    - total_debt : = assume_dte * total_equity
    - total_assets : = equity + debt
    - cash_and_equivalents, capex : 0 si NaN
    """
    df = df_fund.copy()
    df['period'] = df['period'].astype(str)

    # 1) Imputation revenue et net_income via tendance
    if 'revenue' in df.columns:
        df['revenue'] = _fit_yearly_trend_impute(df, 'revenue')
    if 'net_income' in df.columns:
        df['net_income'] = _fit_yearly_trend_impute(df, 'net_income')

    # 2) DPS : forward fill si manquant
    if 'dividend_per_share' in df.columns:
        df['dividend_per_share'] = df['dividend_per_share'].ffill().bfill()

    # 3) EPS
    if 'EPS' not in df.columns:
        df['EPS'] = np.nan
    if {'net_income', 'shares_outstanding'} <= set(df.columns):
        df['EPS'] = df['EPS'].where(df['EPS'].notna(), df['net_income'] / df['shares_outstanding'])

    # 4) PER (lié au prix courant)
    df['PER'] = last_close / df['EPS'].replace(0, np.nan)

    # 5) Dividendes totaux = DPS * nb d'actions si manquant
    if {'dividend_per_share', 'shares_outstanding'} <= set(df.columns):
        if 'dividends_total' not in df.columns:
            df['dividends_total'] = np.nan
        df['dividends_total'] = df['dividends_total'].where(
            df['dividends_total'].notna(),
            df['dividend_per_share'] * df['shares_outstanding']
        )

    # 6) total_equity imputé via ROE supposé si manquant
    if 'total_equity' not in df.columns:
        df['total_equity'] = np.nan
    if 'net_income' in df.columns:
        # éviter division par 0
        roe = max(assume_roe, 1e-6)
        df['total_equity'] = df['total_equity'].where(
            df['total_equity'].notna(),
            df['net_income'] / roe
        )

    # 7) total_debt imputé via D/E supposé
    if 'total_debt' not in df.columns:
        df['total_debt'] = np.nan
    df['total_debt'] = df['total_debt'].where(
        df['total_debt'].notna(),
        assume_dte * df['total_equity']
    )

    # 8) total_assets = equity + debt si manquant
    if 'total_assets' not in df.columns:
        df['total_assets'] = np.nan
    df['total_assets'] = df['total_assets'].where(
        df['total_assets'].notna(),
        df['total_equity'] + df['total_debt']
    )

    # 9) cash & capex -> 0 si NaN (prudence)
    for c in ['cash_and_equivalents', 'capex']:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0.0)

    # 10) Ratios dérivés
    df['Dividend_Yield_%'] = 100 * df['dividend_per_share'] / last_close
    df['ROE_%'] = 100 * df['net_income'] / df['total_equity'].replace(0, np.nan)
    df['Debt_to_Equity'] = df['total_debt'] / df['total_equity'].replace(0, np.nan)
    df['Payout_%'] = 100 * df['dividends_total'] / df['net_income'].replace(0, np.nan)

    # 11) Score fondamental
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

    # 12) Indicateur d'imputation (True si au moins une valeur imputée sur la ligne)
    base_cols = ['revenue','net_income','dividend_per_share','dividends_total','total_equity','total_debt','total_assets','cash_and_equivalents','capex','EPS']
    imputed_flags = []
    for _, row in df.iterrows():
        flag = False
        for c in base_cols:
            # considéré imputé si NaN initialement ? Ici on ne sait plus. On approxime :
            # on marque comme imputée si provient d'une règle évidente (equity, debt, assets, cash/capex=0, DPS ffill pour 2025)
            pass
        imputed_flags.append(np.nan)  # placeholder, optionnel
    df['imputed_info'] = "Auto-complété (ROE≈{:.0f}%, D/E≈{:.2f})".format(assume_roe*100, assume_dte)

    return df

def plot_per(df_ratios: pd.DataFrame) -> go.Figure:
    dfp = df_ratios[['period','PER']].replace([np.inf, -np.inf], np.nan).dropna().sort_values('period')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfp['period'], y=dfp['PER'], mode='lines+markers', name='PER'))
    fig.update_layout(height=350, hovermode='x unified', xaxis_title="Période", yaxis_title="PER")
    return fig

def plot_roe(df_ratios: pd.DataFrame) -> go.Figure:
    dfr = df_ratios[['period','ROE_%']].dropna().sort_values('period')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfr['period'], y=dfr['ROE_%'], mode='lines+markers', name='ROE (%)'))
    fig.update_layout(height=350, hovermode='x unified', xaxis_title="Période", yaxis_title="ROE (%)")
    return fig

def plot_revenue_net_income(df_ratios: pd.DataFrame) -> go.Figure:
    cols = [c for c in ['period','revenue','net_income'] if c in df_ratios.columns]
    dfn = df_ratios[cols].dropna().sort_values('period')
    fig = go.Figure()
    if 'revenue' in dfn.columns:
        fig.add_trace(go.Scatter(x=dfn['period'], y=dfn['revenue'], mode='lines+markers', name="Chiffre d'affaires"))
    if 'net_income' in dfn.columns:
        fig.add_trace(go.Scatter(x=dfn['period'], y=dfn['net_income'], mode='lines+markers', name="Résultat net"))
    fig.update_layout(height=380, hovermode='x unified', xaxis_title="Période", yaxis_title="Montants (FCFA)")
    return fig

def commentaire_auto_points(df_ratios: pd.DataFrame) -> List[str]:
    """Messages courts et clairs sur la dernière période renseignée."""
    notes = []
    if df_ratios.empty or 'period' not in df_ratios.columns:
        return ["Aucune donnée fondamentale disponible."]
    last = df_ratios.sort_values('period').iloc[-1]
    p = str(last.get('period'))

    # EPS
    eps = last.get('EPS', np.nan)
    if pd.notna(eps) and eps > 0:
        notes.append(f"**{p} — EPS positif** : {eps:,.2f} FCFA/action.")
    elif pd.notna(eps):
        notes.append(f"**{p} — EPS faible/négatif** : {eps:,.2f} FCFA/action (à surveiller).")

    # PER
    per = last.get('PER', np.nan)
    if pd.notna(per):
        if 5 <= per <= 20:
            notes.append(f"**{p} — PER** ≈ {per:.1f} (zone raisonnable).")
        elif per < 5:
            notes.append(f"**{p} — PER** ≈ {per:.1f} (potentielle décote, vérifier la qualité du bénéfice).")
        else:
            notes.append(f"**{p} — PER** ≈ {per:.1f} (valorisation tendue).")

    # ROE
    roe = last.get('ROE_%', np.nan)
    if pd.notna(roe):
        if roe >= 15:
            notes.append(f"**{p} — ROE élevé** : {roe:.1f}%.")
        elif roe >= 8:
            notes.append(f"**{p} — ROE correct** : {roe:.1f}%.")
        else:
            notes.append(f"**{p} — ROE faible** : {roe:.1f}%.")

    # D/E
    dte = last.get('Debt_to_Equity', np.nan)
    if pd.notna(dte):
        if dte <= 0.5:
            notes.append(f"**{p} — Endettement maîtrisé** : D/E ≈ {dte:.2f}.")
        elif dte <= 1:
            notes.append(f"**{p} — Endettement modéré** : D/E ≈ {dte:.2f}.")
        else:
            notes.append(f"**{p} — Endettement élevé** : D/E ≈ {dte:.2f}.")

    # Dividend Yield
    dy = last.get('Dividend_Yield_%', np.nan)
    if pd.notna(dy):
        if dy >= 4:
            notes.append(f"**{p} — Rendement dividende attractif** : ≈ {dy:.1f}%.")
        elif dy >= 2:
            notes.append(f"**{p} — Rendement dividende** : ≈ {dy:.1f}%.")

    # Score
    score = last.get('Score_Fondamental_0_10', np.nan)
    if pd.notna(score):
        notes.append(f"**{p} — Score fondamental (0–10)** : **{score:.1f}**.")

    if not notes:
        notes = [f"Données {p} présentes mais incomplètes."]
    return notes

def resume_markdown(df_ratios: pd.DataFrame) -> str:
    """Résumé clair et téléchargeable (Markdown)."""
    lines = ["# Synthèse fondamentale — CFAO CI", ""]
    if df_ratios.empty:
        lines += ["*(Aucune donnée).*"]
        return "\n".join(lines)

    # Aperçu global
    dispo = [c for c in ['revenue','net_income','EPS','PER','ROE_%','Debt_to_Equity','Dividend_Yield_%','Payout_%'] if c in df_ratios.columns]
    lines += ["**Périodes couvertes :** " + ", ".join(df_ratios['period'].astype(str).tolist()),
              "**Ratios disponibles :** " + (", ".join(dispo) if dispo else "aucun"), ""]

    # Dernière période
    last = df_ratios.sort_values('period').iloc[-1]
    p = str(last.get('period'))
    lines += [f"## Dernière période : {p}", ""]
    for msg in commentaire_auto_points(df_ratios):
        lines += [f"- {msg}"]
    lines.append("")

    # Tendances simples (si données multi-périodes)
    if df_ratios['period'].nunique() >= 3:
        dft = df_ratios.sort_values('period')
        def trend(col):
            s = dft[col].dropna()
            if len(s) >= 3:
                return "hausse" if s.iloc[-1] > s.iloc[0] else "baisse" if s.iloc[-1] < s.iloc[0] else "stable"
            return "n/a"
        if 'revenue' in dft.columns:
            lines.append(f"- **Tendance CA** : {trend('revenue')}.")
        if 'net_income' in dft.columns:
            lines.append(f"- **Tendance Résultat net** : {trend('net_income')}.")
        if 'PER' in dft.columns:
            lines.append(f"- **Tendance PER** : {trend('PER')}.")
        if 'ROE_%' in dft.columns:
            lines.append(f"- **Tendance ROE** : {trend('ROE_%')}.")

    lines += ["", "> *Note : interprétation indicative — à croiser avec le contexte macro, la concurrence et les communiqués officiels.*"]
    return "\n".join(lines)

# --------------------------- APP ---------------------------
def main():
    st.sidebar.header("🎛️ Contrôles")

    # ---- Données de PRIX
    uploader = st.sidebar.file_uploader("📥 Importer un CSV (prix) (optionnel)", type=['csv'])
    if uploader is not None:
        df = load_data(uploader)
    else:
        try:
            df = load_data('CFAOCI.csv')
        except Exception:
            st.error("❌ Impossible de charger les données. Importez un CSV ou placez 'CFAOCI.csv' dans le dossier.")
            st.stop()

    st.sidebar.subheader("⏱️ Fenêtre temporelle")
    freq = st.sidebar.selectbox("Fréquence", ['Jour', 'Semaine', 'Mois'], index=0)
    freq_map = {'Jour': 'D', 'Semaine': 'W', 'Mois': 'M'}

    min_date, max_date = df['Date'].min().date(), df['Date'].max().date()
    date_range = st.sidebar.date_input(
        "Plage de dates",
        value=(min_date, max_date),
        min_value=min_date, max_value=max_date
    )
    if isinstance(date_range, tuple):
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    df = resample_ohlcv(df, freq_map[freq])

    st.sidebar.subheader("🧭 Indicateurs")
    indicators = st.sidebar.multiselect(
        "Ajouter au graphique",
        ['MM', 'EMA', 'Bandes de Bollinger', 'RSI', 'MACD'],
        default=['MM', 'RSI']
    )

    with st.sidebar.expander("⚙️ Paramètres indicateurs", expanded=False):
        params = {
            'show_sma': 'MM' in indicators,
            'sma1': st.slider("MM courte", 5, 60, 20, step=1),
            'sma2': st.slider("MM longue", 10, 200, 50, step=1),
            'show_ema': 'EMA' in indicators,
            'ema1': st.slider("EMA", 5, 60, 20, step=1),
            'show_bb': 'Bandes de Bollinger' in indicators,
            'bb_window': st.slider("Bollinger : fenêtre", 10, 60, 20, step=1),
            'bb_std': st.slider("Bollinger : écart-type", 1.0, 3.0, 2.0, step=0.1),
            'show_rsi': 'RSI' in indicators,
            'rsi_window': st.slider("RSI : fenêtre", 5, 30, 14, step=1),
            'show_macd': 'MACD' in indicators,
            'macd_fast': st.slider("MACD : rapide", 5, 20, 12, step=1),
            'macd_slow': st.slider("MACD : lent", 20, 40, 26, step=1),
            'macd_signal': st.slider("MACD : signal", 5, 20, 9, step=1)
        }

    df = add_indicators(df, params)

    st.sidebar.subheader("📊 Type de graphique")
    chart_engine = st.sidebar.radio("Moteur", ['Interactif (Plotly)', 'Statique (Matplotlib)'], index=0)
    chart_type = st.sidebar.radio("Style", ['Ligne', 'Chandelles japonaises'], index=1)

    st.sidebar.subheader("📉 Paramètres risque")
    rf = st.sidebar.number_input("Taux sans risque annuel (%)", value=2.0, step=0.5)

    # ===================== MÉTRIQUES TECHNIQUES =====================
    metrics = performance_metrics(df, rf_annual_pct=rf)
    st.subheader("📈 Métriques Principales")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("💰 Prix Actuel", f"{metrics['current_price']:.0f} FCFA", help="Dernière clôture")
    c2.metric("📊 Rendement Total", f"{metrics['total_return']:.2f}%")
    c3.metric("📅 Rendement Annualisé", f"{metrics['annualized_return']:.2f}%")
    c4.metric("⚡ Volatilité (ann.)", f"{metrics['volatility']:.2f}%")
    c5.metric("📉 Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
    c6.metric("📐 Sharpe", f"{metrics['sharpe']:.2f}")

    st.info(f"📅 **Dernière mise à jour:** {metrics['last_update']} | 📊 **Nombre de sessions:** {len(df)} | 📦 **Volume Moyen:** {metrics['avg_volume']:,.0f}")
    st.markdown("---")

    # ===================== ANALYSE TECHNIQUE =====================
    st.subheader("📈 Analyse Technique")
    tab1, tab2, tab3 = st.tabs(["📈 Graphique principal", "🎯 RSI & MACD", "📋 Données"])

    with tab1:
        if chart_engine.startswith('Interactif'):
            main_fig = plotly_main_chart(df, 'Chandelles japonaises' if chart_type == 'Chandelles japonaises' else 'Ligne', params)
            st.plotly_chart(main_fig, use_container_width=True, config={"displaylogo": False})
        else:
            if chart_type == 'Chandelles japonaises':
                fig, ax = plt.subplots(figsize=(12, 6))
                for date, o, h, l, c in zip(df['Date'], df['Open'], df['High'], df['Low'], df['Close']):
                    color = 'green' if c >= o else 'red'
                    ax.plot([date, date], [l, h], color='black', linewidth=1)
                    ax.bar(date, abs(c - o), bottom=min(o, c), width=0.8, color=color, alpha=0.8, edgecolor='black', linewidth=0.3)
                if params.get('show_sma'):
                    ax.plot(df['Date'], df['SMA_1'], label=f"MM {params['sma1']}", linestyle='--')
                    ax.plot(df['Date'], df['SMA_2'], label=f"MM {params['sma2']}", linestyle='--')
                if params.get('show_ema'):
                    ax.plot(df['Date'], df['EMA_1'], label=f"EMA {params['ema1']}")
                if params.get('show_bb'):
                    ax.plot(df['Date'], df['BB_M'], label="BB moyenne", alpha=0.8)
                    ax.plot(df['Date'], df['BB_U'], label="BB sup", alpha=0.6)
                    ax.plot(df['Date'], df['BB_L'], label="BB inf", alpha=0.6)
                ax.set_title('Chandelles CFAOCI'); ax.set_ylabel("Prix (FCFA)"); ax.legend(); ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y')); plt.xticks(rotation=45); st.pyplot(fig, clear_figure=True)
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df['Date'], df['Close'], label='Clôture', linewidth=2)
                if params.get('show_sma'):
                    ax.plot(df['Date'], df['SMA_1'], label=f"MM {params['sma1']}", linestyle='--')
                    ax.plot(df['Date'], df['SMA_2'], label=f"MM {params['sma2']}", linestyle='--')
                if params.get('show_ema'):
                    ax.plot(df['Date'], df['EMA_1'], label=f"EMA {params['ema1']}")
                if params.get('show_bb'):
                    ax.plot(df['Date'], df['BB_M'], label="BB moyenne", alpha=0.8)
                    ax.fill_between(df['Date'], df['BB_L'], df['BB_U'], alpha=0.1, label="Bollinger")
                ax.set_title('Clôture CFAOCI'); ax.set_ylabel("Prix (FCFA)"); ax.legend(); ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y')); plt.xticks(rotation=45); st.pyplot(fig, clear_figure=True)

    with tab2:
        colA, colB = st.columns(2)
        if 'RSI' in indicators:
            with colA:
                st.subheader("🎯 RSI")
                st.plotly_chart(plotly_rsi_chart(df), use_container_width=True, config={"displaylogo": False})
        if 'MACD' in indicators:
            with colB:
                st.subheader("📊 MACD")
                st.plotly_chart(plotly_macd_chart(df), use_container_width=True, config={"displaylogo": False})

    with tab3:
        st.subheader("📋 Données Récentes")
        display_df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Variation']].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%d/%m/%Y')
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Télécharger le CSV filtré", csv, file_name="CFAOCI_filtre.csv", mime="text/csv")

    # ===================== ANALYSE AUTO (TECHNIQUE) =====================
    st.markdown("---")
    st.subheader("🤖 Analyse Technique Automatique")
    latest = df.iloc[-1]
    notes = []

    if 'RSI' in df.columns:
        rsi_value = float(latest['RSI'])
        if rsi_value > 70:
            notes.append(f"⚠️ **Signal RSI:** Surachat détecté (RSI: {rsi_value:.1f})")
        elif rsi_value < 30:
            notes.append(f"✅ **Signal RSI:** Zone de survente (RSI: {rsi_value:.1f})")
        else:
            notes.append(f"ℹ️ **Signal RSI:** Zone neutre (RSI: {rsi_value:.1f})")

    if 'SMA_1' in df.columns and 'SMA_2' in df.columns:
        cond_up = latest['Close'] > latest['SMA_1'] > latest['SMA_2']
        cond_down = latest['Close'] < latest['SMA_1'] < latest['SMA_2']
        if cond_up:
            notes.append("🚀 **Tendance:** Haussière forte (Prix > MM courte > MM longue)")
        elif cond_down:
            notes.append("📉 **Tendance:** Baissière forte (Prix < MM courte < MM longue)")
        else:
            notes.append("⚖️ **Tendance:** Neutre/Consolidation (MM croisées ou proches)")

    colL, colR = st.columns(2)
    with colL:
        for n in notes:
            st.write("- " + n)
    with colR:
        st.info(f"💹 **Prix maximum:** {df['Close'].max():.0f} FCFA | 📉 **Prix minimum:** {df['Close'].min():.0f} FCFA")

    # ===================== ANALYSE FONDAMENTALE (auto-complétion) =====================
    st.markdown("---")
    st.subheader("📚 Analyse Fondamentale — CFAO CI")

    st.sidebar.subheader("📥 Données fondamentales")
    fund_uploader = st.sidebar.file_uploader("Importer un CSV fondamentaux (facultatif)", type=['csv'], key="fund_csv")

    if fund_uploader is not None:
        try:
            df_fund = load_fundamentals(fund_uploader)
        except Exception:
            st.warning("⚠️ Fichier fondamentaux illisible. Utilisation des données intégrées par défaut.")
            df_fund = fundamentals_default_df()
    else:
        df_fund = fundamentals_default_df()

    with st.sidebar.expander("🛠️ Auto-compléter les valeurs manquantes", expanded=True):
        assume_roe_pct = st.slider("ROE supposé pour imputation (%)", min_value=5, max_value=25, value=12, step=1)
        assume_dte = st.slider("Dette / Capitaux propres supposé (D/E)", min_value=0.0, max_value=2.0, value=0.60, step=0.05)

    last_close = float(metrics['current_price'])
    df_ratios = impute_fundamentals(
        df_fund,
        assume_roe=assume_roe_pct/100.0,
        assume_dte=assume_dte,
        last_close=last_close
    )

    tabF1, tabF2, tabF3 = st.tabs(["📊 Ratios & Score (imputés)", "📈 Graphiques", "🧠 Commentaire auto & Téléchargements"])

    with tabF1:
        cols_show = [c for c in [
            'period', 'revenue', 'net_income', 'EPS', 'PER', 'ROE_%',
            'Debt_to_Equity', 'Dividend_Yield_%', 'Payout_%', 'dividend_per_share',
            'dividends_total', 'total_equity', 'total_debt', 'total_assets', 'cash_and_equivalents', 'capex',
            'Score_Fondamental_0_10', 'imputed_info'
        ] if c in df_ratios.columns]
        st.dataframe(df_ratios[cols_show], use_container_width=True)

        # Export CSV des ratios (après imputation)
        csv_ratios = df_ratios.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Télécharger les ratios imputés (CSV)", csv_ratios, file_name="CFAOCI_fondamentaux_imputes.csv", mime="text/csv")

    with tabF2:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_revenue_net_income(df_ratios), use_container_width=True, config={"displaylogo": False})
        with col2:
            st.plotly_chart(plot_per(df_ratios), use_container_width=True, config={"displaylogo": False})
        st.plotly_chart(plot_roe(df_ratios), use_container_width=True, config={"displaylogo": False})

    with tabF3:
        st.markdown("### Résumé clair (auto, après imputation)")
        for line in commentaire_auto_points(df_ratios):
            st.write("- " + line)

        # Résumé téléchargeable (Markdown)
        md_text = resume_markdown(df_ratios)
        st.markdown("---")
        st.markdown("#### Télécharger le résumé (Markdown)")
        st.download_button(
            "📝 Télécharger le résumé (.md)",
            data=md_text.encode('utf-8'),
            file_name="CFAOCI_resume_fondamental.md",
            mime="text/markdown"
        )

        st.caption(
            "Interprétation indicative. Les valeurs imputées utilisent vos hypothèses (ROE, D/E). Remplacez-les par les chiffres officiels dès que disponibles."
        )

    # --------------------- PIED ---------------------
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p><strong>Dashboard CFAOCI - Données BRVM</strong></p>
            <p>Analyse technique & fondamentale — Développé avec ❤️</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
