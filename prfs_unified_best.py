"""
Pakistan Revenue Forecasting System (PRFS) - Unified Best Version
==================================================================
Combines the CORRECT model logic from app1.py with the BEAUTIFUL UI from app2.py

Features:
- Correct model calculations and data transforms from app1
- Executive design system and professional dashboard from app2
- Editable data table with auto-validation
- Beautiful KPI tiles, charts, and insights panel

Run:
    streamlit run prfs_unified_best.py
"""
from __future__ import annotations

import sys, os

# Ensure package is importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(r'C:\Users\LENOVO\Downloads\Pakistan_Income_Tax_Slabs_app\PRFS_II')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import traceback
import hashlib as _hashlib

# ── Local imports from app1 (CORRECT MODEL LOGIC) ────────────────────────────
from prfs_unified.data_io import (
    load_tax_data,
    prepare_transforms,
    load_buoyancy,
    load_multimodel_assets,
)
from prfs_unified.scenario_inputs import render_sidebar, TAX_LABELS, MODEL_LABELS
from prfs_unified.plots import forecast_plot, forecast_table
from prfs_unified.buoyancy_benchmark import render_benchmark
from prfs_unified.utils import (
    diagnostics_ardl,
    diagnostics_arimax,
    coef_table_ardl,
    coef_table_arimax,
    coef_table_enet,
)
from prfs_unified.adapters import multimodel_adapter as mm
from prfs_unified.adapters import dynamic_adapter as dyn
from prfs_unified.mapping import build_multimodel_future_exog_from_dynamic

# ═════════════════════════════════════════════════════════════════════════
# Page config
# ═════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Pakistan Revenue Forecasting System (PRFS)",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ═════════════════════════════════════════════════════════════════════════
# EXECUTIVE DESIGN SYSTEM (from app2 - BEAUTIFUL UI)
# ═════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --primary-600: #2563EB;
    --primary-700: #1D4ED8;
    --accent-purple: #8B5CF6;
    --accent-teal: #14B8A6;
    --accent-orange: #F59E0B;
    --accent-rose: #F43F5E;

    --gray-50: #FAFBFC;
    --gray-100: #F4F6F8;
    --gray-600: #4B5563;
    --gray-800: #1F2937;

    --text-primary: #1F2937;
    --text-secondary: #4B5563;
    --text-tertiary: #6B7280;

    --surface-primary: #FFFFFF;
    --surface-secondary: #FAFBFC;
    --surface-elevated: #FFFFFF;

    --bg-page: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    --bg-sidebar: linear-gradient(180deg, #e3ebe6 0%, #cfdcd4 100%);

    --border-light: #F0F3F6;
    --border-medium: #E5E9ED;
    --border-strong: #D1D8DE;

    --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.06);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.10);

    --radius-md: 10px;
    --radius-lg: 14px;
    --radius-xl: 20px;

    --space-4: 1rem;
    --space-6: 1.5rem;
    --space-8: 2rem;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, .stApp, .main, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-page) !important;
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

.main .block-container {
    padding: 0rem 2.5rem 3rem;
    max-width: 1800px;
    margin: 0 auto;
    padding-top: 1rem !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-medium);
    box-shadow: var(--shadow-lg);
}

/* KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: var(--space-6);
    margin: var(--space-8) 0;
}

.kpi-card {
    background: var(--surface-elevated);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-lg);
    padding: var(--space-6);
    box-shadow: var(--shadow-md);
    transition: all 0.3s ease;
}

.kpi-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}

.kpi-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
}

.kpi-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    flex-shrink: 0;
}

.kpi-icon.blue { background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%); }
.kpi-icon.purple { background: linear-gradient(135deg, #EDE9FE 0%, #DDD6FE 100%); }
.kpi-icon.teal { background: linear-gradient(135deg, #CCFBF1 0%, #99F6E4 100%); }
.kpi-icon.orange { background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%); }

.kpi-label {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi-value {
    font-size: 32px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
    margin: 8px 0;
}

.kpi-change {
    font-size: 14px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 6px;
    display: inline-block;
}

.kpi-change.positive {
    color: #047857;
    background: #ECFDF5;
}

.kpi-change.negative {
    color: #991B1B;
    background: #FEE2E2;
}

/* Insight Panel */
.insight-panel {
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
    border: 1px solid #BFDBFE;
    border-radius: var(--radius-lg);
    padding: var(--space-6);
    margin: var(--space-8) 0;
    box-shadow: var(--shadow-md);
}

.insight-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}

.insight-icon {
    font-size: 28px;
}

.insight-title {
    font-size: 18px;
    font-weight: 700;
    color: #1E40AF;
}

.insight-content {
    font-size: 15px;
    line-height: 1.7;
    color: #1F2937;
}

/* Content Sections */
.content-section {
    background: var(--surface-elevated);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-lg);
    padding: var(--space-8);
    margin: var(--space-6) 0;
    box-shadow: var(--shadow-sm);
}

.section-header {
    margin-bottom: var(--space-6);
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 8px;
}

.section-subtitle {
    font-size: 14px;
    color: var(--text-secondary);
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #0a3d62, #1e3799);
    padding: 28px 32px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: var(--shadow-lg);
}

.app-title {
    color: #fff;
    margin: 0;
    font-family: 'Inter', sans-serif;
    font-size: 32px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ── Header (Beautiful) ────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">Pakistan Revenue Forecasting System (PRFS)</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════════════════
# Load / initialise data (CORRECT LOGIC from app1)
# ═════════════════════════════════════════════════════════════════════════
try:
    if "user_df" not in st.session_state:
        _df_file = load_tax_data()
        _df_file = prepare_transforms(_df_file)
        st.session_state["user_df"] = _df_file
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

df_raw = st.session_state["user_df"]

# ── Load multi-model assets (CORRECT from app1) ──────────────────────────
bundle, meta, _df_hist_from_file = load_multimodel_assets()

# Rebuild df_hist from current user data
df_hist = df_raw.copy()

# Re-compute ENet residuals (CORRECT from app1)
if bundle is not None:
    for head, b in bundle["models"].items():
        if "enet" in b:
            model      = b["enet"]["model"]
            feat_cols  = b["enet"]["feature_cols"]
            y_name     = b["spec"]["y"]
            train_resids = []
            valid_hist = df_hist.dropna(subset=[y_name]).index[2:] if y_name in df_hist.columns else []
            for t in valid_hist:
                try:
                    row = {}
                    for c in feat_cols:
                        if c.endswith("_L0"):
                            base = c[:-3]
                            row[c] = df_hist.loc[t, base] if base in df_hist.columns else 0.0
                        elif "_L" in c:
                            parts = c.rsplit("_L", 1)
                            base, lag = parts[0], int(parts[1])
                            row[c] = df_hist.shift(lag).loc[t, base] if base in df_hist.columns else 0.0
                        else:
                            row[c] = df_hist.loc[t, c] if c in df_hist.columns else 0.0
                    row_df = pd.DataFrame([row], columns=feat_cols).fillna(0)
                    pred   = float(model.predict(row_df)[0])
                    if y_name in df_hist.columns:
                        train_resids.append(df_hist.loc[t, y_name] - pred)
                except Exception:
                    continue
            b["enet"]["residuals"] = train_resids if train_resids else [0.0]

buoy_data = load_buoyancy()

multimodel_ok = bundle is not None
dynamic_ok    = dyn.is_available()
perf          = mm.perf_table(meta) if meta else None

# ── Dynamic year metadata (CORRECT from app1) ─────────────────────────────
_max_year = int(df_raw.index.max().year) if hasattr(df_raw.index, "year") else int(str(df_raw.index.max())[:4])
_min_year = int(df_raw.index.min().year) if hasattr(df_raw.index, "min") else int(str(df_raw.index.min())[:4])

# ── Data version for cache-busting (CORRECT from app1) ────────────────────
_data_version = _hashlib.md5(
    pd.util.hash_pandas_object(df_raw, index=True).values.tobytes()
).hexdigest()[:12]

# ═════════════════════════════════════════════════════════════════════════
# Sidebar (CORRECT from app1)
# ═════════════════════════════════════════════════════════════════════════
cfg = render_sidebar(
    perf=perf,
    dynamic_available=dynamic_ok,
    multimodel_available=multimodel_ok,
)

# Show data coverage in sidebar
st.sidebar.info(f"📅 Data Coverage: FY{_min_year} – FY{_max_year}")
st.sidebar.caption(f"🔮 Forecast year: FY{_max_year + 1}+")

head        = cfg["head"]
horizon     = cfg["horizon"]
n_sims      = cfg["n_sims"]
targets     = cfg["targets"]
elasticities = cfg["elasticities"]
covid_on    = cfg["covid_on"]
regime_on   = cfg["regime_on"]

# ═════════════════════════════════════════════════════════════════════════
# Resolve chosen model (CORRECT from app1)
# ═════════════════════════════════════════════════════════════════════════
chosen = cfg["model_choice"]
is_mm  = chosen in ("ardl", "arimax", "enet")
default_model_label = MODEL_LABELS.get(chosen, chosen.upper())

# ═════════════════════════════════════════════════════════════════════════
# Generate forecasts (CORRECT LOGIC from app1)
# ═════════════════════════════════════════════════════════════════════════
fore_head  = None
fore_total = None
exog_future = None
dyn_results = None

try:
    if is_mm:
        # Multi-model forecast (CORRECT from app1)
        fore_head, exog_future = mm.forecast_head(
            bundle, meta, df_hist,
            head, chosen, horizon, n_sims,
            targets, elasticities, covid_on, regime_on,
            data_version=_data_version,
        )
        fore_total = mm.forecast_total(
            bundle, meta, df_hist,
            chosen, horizon, n_sims,
            targets, elasticities, covid_on, regime_on,
            data_version=_data_version,
        )
    else:
        # Dynamic forecast (CORRECT from app1)
        dyn_results = dyn.run_dynamic_system(
            df_hist, horizon, targets, elasticities,
            covid_on=covid_on, regime_on=regime_on,
        )
        if dyn_results:
            fore_head  = dyn_results["forecasts"].get(head)
            fore_total = dyn_results["aggregates"]["total_revenue"]
            exog_future = build_multimodel_future_exog_from_dynamic(dyn_results, horizon)
except Exception as e:
    st.error(f"Forecast generation error: {e}")
    traceback.print_exc()

# ═════════════════════════════════════════════════════════════════════════
# Calculate metrics for KPI tiles
# ═════════════════════════════════════════════════════════════════════════
insight_cat_val = 0.0
insight_cat_growth = 0.0
total_fore_2027 = 0.0
total_cagr_final = 0.0
mae_display = "N/A"

if fore_head is not None and len(fore_head) > 0:
    # Extract scalar from DataFrame if needed (usually 1st column is point forecast)
    insight_cat_val = float(fore_head.iloc[0, 0] if hasattr(fore_head, 'columns') else fore_head.iloc[0]) / 1000
    
    # Get last historical value for growth calculation
    if head in df_hist.columns:
        last_hist = df_hist[head].dropna().iloc[-1] if len(df_hist[head].dropna()) > 0 else 1.0
        insight_cat_growth = float(((insight_cat_val * 1000 - last_hist) / last_hist * 100)) if last_hist != 0 else 0.0

if fore_total is not None and len(fore_total) > 0:
    total_fore_2027 = float(fore_total.iloc[0, 0] if hasattr(fore_total, 'columns') else fore_total.iloc[0]) / 1000
    
    # Calculate CAGR
    if "total" in df_hist.columns:
        last_total = df_hist["total"].dropna().iloc[-1] if len(df_hist["total"].dropna()) > 0 else 1.0
        total_cagr_final = float(((total_fore_2027 * 1000 - last_total) / last_total * 100)) if last_total != 0 else 0.0

# Get MAE from performance metrics
if perf is not None and head in perf.index:
    try:
        mae_display = f"{perf.loc[head, 'mae']:.2f}" if 'mae' in perf.columns else "N/A"
    except:
        mae_display = "N/A"

# ═════════════════════════════════════════════════════════════════════════
# KPI TILES (BEAUTIFUL UI from app2)
# ═════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-icon blue">📊</div>
            <div class="kpi-label">{TAX_LABELS.get(head, head)} Forecast</div>
        </div>
        <div class="kpi-value">Rs. {insight_cat_val:,.0f}B</div>
        <div class="kpi-change {'positive' if insight_cat_growth >= 0 else 'negative'}">
            {insight_cat_growth:+.1f}% vs Current FY
        </div>
    </div>
    
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-icon purple">💰</div>
            <div class="kpi-label">Total Revenue (FY{_max_year + 1})</div>
        </div>
        <div class="kpi-value">Rs. {total_fore_2027:,.0f}B</div>
        <div class="kpi-change {'positive' if total_cagr_final >= 0 else 'negative'}">
            {total_cagr_final:+.1f}% Growth
        </div>
    </div>
    
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-icon teal">🎯</div>
            <div class="kpi-label">Model Accuracy</div>
        </div>
        <div class="kpi-value">{mae_display}</div>
        <div class="kpi-change positive">MAE (Out-of-sample)</div>
    </div>
    
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-icon orange">🔮</div>
            <div class="kpi-label">Forecast Model</div>
        </div>
        <div class="kpi-value">{default_model_label}</div>
        <div class="kpi-change positive">Selected Engine</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Forecast", "📈 Diagnostics", "📋 Coefficients",
    "🎯 Benchmark", "🔬 Methodology", "✏️ Data Preview", "📄 Reports"
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — Forecast (with CORRECT data and BEAUTIFUL charts)
# ─────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            <div class="section-title">Revenue Forecast</div>
            <div class="section-subtitle">Model projections with confidence intervals</div>
        </div>
    """, unsafe_allow_html=True)
    
    if fore_head is not None and len(fore_head) > 0:
        # Use the CORRECT forecast_plot function from app1
        fig = forecast_plot(df_hist, fore_head, head, _max_year)
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast table
        st.subheader("Forecast Table")
        tbl = forecast_table(fore_head, _max_year + 1)
        st.dataframe(tbl, use_container_width=True)
    else:
        st.warning("No forecast data available. Check model selection and inputs.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — Diagnostics (CORRECT logic from app1)
# ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            <div class="section-title">Model Diagnostics</div>
            <div class="section-subtitle">Statistical tests and residual analysis</div>
        </div>
    """, unsafe_allow_html=True)
    
    if is_mm and bundle:
        model_data = bundle["models"][head].get(chosen)
        if model_data and "res" in model_data:
            res = model_data["res"]
            if chosen == "ardl":
                metrics = diagnostics_ardl(res)
            else:
                metrics = diagnostics_arimax(res)
            st.json(metrics)
        elif chosen == "enet":
            st.info("Elastic Net diagnostics: Built using regularization path and cross-validation.")
            if model_data and "residuals" in model_data:
                st.write(f"RMS Residual: {np.sqrt(np.mean(np.square(model_data['residuals']))):.4f}")
    else:
        st.info("Diagnostics available for multi-model engines only.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — Coefficients (CORRECT logic from app1)
# ─────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            <div class="section-title">Model Coefficients</div>
            <div class="section-subtitle">Estimated parameters and significance levels</div>
        </div>
    """, unsafe_allow_html=True)
    
    if is_mm and bundle:
        model_data = bundle["models"][head].get(chosen)
        if model_data:
            if chosen == "ardl" and "res" in model_data:
                st.dataframe(coef_table_ardl(model_data["res"]), use_container_width=True)
            elif chosen == "arimax" and "res" in model_data:
                st.dataframe(coef_table_arimax(model_data["res"]), use_container_width=True)
            elif chosen == "enet":
                st.dataframe(coef_table_enet(bundle["models"][head]), use_container_width=True)
    else:
        st.info("Coefficient tables available for multi-model engines only.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 4 — Benchmark (CORRECT logic from app1)
# ─────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            <div class="section-title">Buoyancy Benchmark</div>
            <div class="section-subtitle">Tax-to-GDP elasticity analysis</div>
        </div>
    """, unsafe_allow_html=True)
    
    if buoy_data:
        render_benchmark(buoy_data, fore_head, fore_total, head)
    else:
        st.warning("Buoyancy data not available.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 5 — Methodology (from app1)
# ─────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            <div class="section-title">Methodology</div>
            <div class="section-subtitle">Model specifications and econometric foundations</div>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📐 ARDL — Autoregressive Distributed Lag Model"):
        st.markdown("""
**Reference:** Pesaran & Shin (1999), *Econometrica*

The ARDL model captures both short-run dynamics and long-run equilibrium relationships 
between tax revenue and its structural determinants.

**Specification:**
$$\\log T_t = c + \\rho \\log T_{t-1} + \\beta^* \\log \\hat{X}_t + \\delta^* D_t + u_t$$

where:
- $T_t$ = Tax revenue
- $\\hat{X}_t$ = Fitted structural base (from Stage-1)
- $D_t$ = Policy regime dummies
- $\\rho$ = Persistence coefficient
- $\\beta^*$ = Short-run elasticity

**Elasticity Recovery:**

| Horizon | Formula | Interpretation |
|---------|---------|----------------|
| Short-Run | $\\hat{\\beta}^*$ | Elasticity of revenue to base within the fiscal year |
| Long-Run | $\\hat{\\beta}^* / (1 - \\hat{\\rho})$ | Permanent elasticity after full convergence |
""")

    with st.expander("🧮 Elastic Net — Penalised Regression"):
        st.markdown("""
**Reference:** Zou & Hastie (2005), *JRSS-B 67(2)*

Regularised regression combining L1 (LASSO) and L2 (Ridge) penalties.

**Objective function:**
$$\\hat{\\beta} = \\arg\\min_{\\beta} \\left[ \\frac{1}{2T} \\|y - X\\beta\\|_2^2 + \\alpha \\left( \\frac{1-\\rho_{L1}}{2} \\|\\beta\\|_2^2 + \\rho_{L1} \\|\\beta\\|_1 \\right) \\right]$$

**Key properties:**
- Consistent variable selection under multicollinearity
- Groups correlated regressors (unlike LASSO)
- Hyper-parameters tuned by rolling-origin CV
""")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 6 — Data Preview (EDITABLE with CORRECT validation from app1)
# ─────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            <div class="section-title">Data Preview</div>
            <div class="section-subtitle">Historical data (editable) and budget estimates</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.info(
        "✏️ **Historical Data is fully editable.**  \n"
        "- Modify any cell or **Add a new row** at the bottom for future years.  \n"
        "- All models, graphs, and forecasts are powered *exclusively* by this table.  \n"
        "- The system will **automatically re-run** when you make changes."
    )

    # Build the editable frame (CORRECT from app1)
    _log_cols = [c for c in df_hist.columns if c.startswith("log_")]
    _edit_cols = [c for c in df_hist.columns if c not in _log_cols]

    if "editor_df" not in st.session_state:
        st.session_state["editor_df"] = df_hist[_edit_cols].reset_index(drop=True)

    _col_cfg: dict = {}
    for col in st.session_state["editor_df"].columns:
        if col == "year_end":
            _col_cfg[col] = st.column_config.NumberColumn(
                "Year End", required=True, min_value=1990, max_value=2100, step=1, format="%d"
            )
        elif col in ("covid", "regime", "step_2024", "dummy_2024", "dummy_2025",
                     "dummy_1995", "dummy_1996", "dummy_2002", "dummy_2003"):
            _col_cfg[col] = st.column_config.CheckboxColumn(col)
        else:
            _col_cfg[col] = st.column_config.NumberColumn(col, format="%.2f")

    edited_df = st.data_editor(
        st.session_state["editor_df"],
        num_rows="dynamic",
        use_container_width=True,
        column_config=_col_cfg,
        key="hist_data_editor",
    )

    # Auto-Apply logic (CORRECT validation from app1)
    _main_changed = not edited_df.equals(st.session_state["editor_df"])
    
    if _main_changed:
        _err = None

        # 1. Drop rows where year_end is blank
        _work = edited_df.dropna(subset=["year_end"]).copy()
        if _work.empty:
            _err = "No rows with a valid year_end found."
        else:
            # 2. Coerce year_end to int
            try:
                _work["year_end"] = _work["year_end"].astype(int)
            except Exception:
                _err = "year_end must be an integer (e.g. 2026)."

        if _err is None:
            # 3. Check uniqueness
            _dupes = _work[_work.duplicated("year_end", keep=False)]["year_end"].unique().tolist()
            if _dupes:
                _err = f"Duplicate year(s) detected: {sorted(_dupes)}. Each year must appear exactly once."

        if _err is None:
            # 4. Validate required columns
            _required = ["dt", "gst", "fed", "customs", "gdp", "imports", "exrate"]
            _missing  = [c for c in _required if c not in _work.columns]
            if _missing:
                _err = f"Required columns missing: {_missing}"

        if _err:
            st.error(f"❌ {_err}")
        else:
            # 5. Sort and set PeriodIndex
            _work = _work.sort_values("year_end").reset_index(drop=True)
            _work.index = pd.PeriodIndex(_work["year_end"], freq="Y")

            # 6. Re-apply transforms (CORRECT from app1)
            _work = prepare_transforms(_work)

            # 7. Persist in session_state
            st.session_state["user_df"]   = _work
            st.session_state["editor_df"] = _work[
                [c for c in _work.columns if not c.startswith("log_")]
            ].reset_index(drop=True)

            # 8. Clear forecast caches
            mm.get_cached_forecast.clear()

            # 9. Clear DSM pipeline
            st.session_state.pop("dyn_pipeline", None)

            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 7 — Reports
# ─────────────────────────────────────────────────────────────────────────
with tab7:
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            <div class="section-title">Reports</div>
            <div class="section-subtitle">Export and documentation</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.info("Report generation coming soon. You can export data from the forecast table above.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════
# INSIGHTS PANEL (BEAUTIFUL from app2 with CORRECT data from app1)
# ═════════════════════════════════════════════════════════════════════════
if fore_total is not None and len(fore_total) > 0 and fore_head is not None and len(fore_head) > 0:
    st.markdown(f"""
    <div class="insight-panel">
        <div class="insight-header">
            <div class="insight-icon">💡</div>
            <div class="insight-title">Executive Insights (FY {_max_year + 1})</div>
        </div>
        <div class="insight-content">
            The <strong>{default_model_label}</strong> model projects
            <strong>{TAX_LABELS[head]}</strong> revenue at <strong>Rs. {insight_cat_val:,.2f} Billion</strong>
            for FY {_max_year + 1}, representing a <strong>{insight_cat_growth:+.1f}%</strong> change
            from the previous fiscal year. Aggregate revenue across all tax heads is expected to reach
            <strong>Rs. {total_fore_2027:,.2f} Billion</strong>, with a growth rate of
            <strong>{total_cagr_final:+.2f}%</strong>. The model achieved an out-of-sample error (MAE) of
            <strong>{mae_display}</strong> during validation.
        </div>
    </div>
    """, unsafe_allow_html=True)
