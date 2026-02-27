# app.py — ValuationSuite USIL (PYG)
# Dashboard premium + DCF + Monte Carlo + One‑Pager (TXT/PDF)
# ✅ Sin dataclasses (evita errores de orden de campos en Python 3.13)
# ✅ Sin matplotlib (Streamlit Cloud friendly)
# ✅ PYG fijo, Nominal fijo, D/E contable fijo, sin semilla expuesta
# ✅ Comité por defecto: VAN base >0, P(VAN<0)≤20%, P50>0, P5>0
# ✅ Contable Año 1 → FCF Año 1 (ventas, costos, dep, CAPEX, ΔCT por componentes)
# ✅ Flujos 1..N explícitos + TV (g∞) explícito
# ✅ Incluye histograma de distribución en dashboard + explicación ejecutiva ampliada
# ✅ PDF one‑pager: KPIs + mini histograma (bins) + resumen

import io
import textwrap
from datetime import date

import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ReportLab (PDF)
REPORTLAB_OK = True
try:
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
except Exception:
    REPORTLAB_OK = False

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="ValuationSuite USIL — One‑Pager (PYG)", layout="wide")

CURRENCY = "Gs."
MIN_SPREAD = 0.005          # WACC debe ser > g∞ + 0.5%
DEFAULT_SIMS = 15000
MAX_PROB_NEG = 0.20         # Comité: P(VAN<0) ≤ 20%
HIST_BINS = 40

# ----------------------------
# Utils
# ----------------------------
def fmt_gs(x: float) -> str:
    try:
        return f"{CURRENCY} {x:,.0f}".replace(",", ".")
    except Exception:
        return f"{CURRENCY} {x}"

def fmt_pct(x: float | None) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "N/A"

def safe_irr(cashflows: np.ndarray):
    try:
        r = float(npf.irr(cashflows))
        if np.isnan(r) or np.isinf(r):
            return None
        if r < -0.99 or r > 5.0:
            return None
        return r
    except Exception:
        return None

def payback_simple(cashflows: np.ndarray):
    cum = 0.0
    for t in range(len(cashflows)):
        cum += cashflows[t]
        if cum >= 0 and t > 0:
            prev = cum - cashflows[t]
            if cashflows[t] == 0:
                return float(t)
            frac = (0 - prev) / cashflows[t]
            return float((t - 1) + frac)
    return None

def payback_discounted(cashflows: np.ndarray, wacc: float):
    cum = 0.0
    for t in range(len(cashflows)):
        pv = cashflows[t] / ((1 + wacc) ** t)
        cum += pv
        if cum >= 0 and t > 0:
            prev = cum - pv
            if pv == 0:
                return float(t)
            frac = (0 - prev) / pv
            return float((t - 1) + frac)
    return None

def committee_verdict(npv_base: float, prob_neg: float, p5: float, p50: float):
    checks = {
        "VAN base > 0": npv_base > 0,
        "P(VAN<0) ≤ 20%": prob_neg <= MAX_PROB_NEG,
        "P50(VAN) > 0": p50 > 0,
        "P5(VAN) > 0": p5 > 0,
    }
    ok = sum(checks.values())
    if ok == len(checks):
        return "APROBADO", "El proyecto cumple criterios conservadores: creación de valor y downside controlado bajo incertidumbre razonable.", checks
    if ok == 0:
        return "RECHAZADO", "El proyecto no cumple criterios mínimos del comité. Requiere revisión integral de supuestos y mitigaciones.", checks
    return "OBSERVADO", "El proyecto presenta potencial; sin embargo, requiere evidencia/mitigaciones sobre supuestos críticos antes de aprobación.", checks

def cvar_left_tail(x: np.ndarray, alpha: float = 0.05) -> float:
    """CVaR (Expected Shortfall) del 5% peor: promedio de la cola izquierda."""
    if x.size == 0:
        return np.nan
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    if tail.size == 0:
        return float(q)
    return float(np.mean(tail))

# ----------------------------
# Styling
# ----------------------------
st.markdown("""
<style>
:root{
  --bg:#070b14;
  --card:#0f1628;
  --card2:#0b1020;
  --border:#1c2a45;
  --txt:#e9eefc;
  --muted:#a7b4d6;
  --accent:#4aa3ff;
  --good:#29d07f;
  --warn:#f6c343;
  --bad:#ff5d6c;
}
html, body, [class*="css"]  { background: var(--bg); }
.block-container {padding-top: 1.0rem; padding-bottom: 1.5rem; max-width: 1250px;}
h1,h2,h3{color:var(--txt);}
.small-muted{color:var(--muted); font-size:0.90rem;}
.card{
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border:1px solid var(--border);
  border-radius:18px;
  padding:16px 16px;
}
.kpi{
  background: radial-gradient(1200px 180px at 30% 0%, rgba(74,163,255,0.18), rgba(0,0,0,0)) , var(--card2);
  border:1px solid var(--border);
  border-radius:16px;
  padding:14px 14px;
  height: 98px;
  overflow:hidden;
}
.kpi .label{color:var(--muted); font-size:0.85rem; margin-bottom:6px;}
.kpi .value{color:var(--txt); font-size:1.20rem; font-weight:750; line-height:1.08; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.kpi .sub{color:var(--muted); font-size:0.80rem; margin-top:6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.pill{
  display:inline-block;
  padding:5px 12px;
  border-radius:999px;
  font-size:0.85rem;
  border:1px solid var(--border);
  color:var(--txt);
  background: rgba(255,255,255,0.03);
}
.pill.good{border-color: rgba(41,208,127,0.55); background: rgba(41,208,127,0.10);}
.pill.warn{border-color: rgba(246,195,67,0.55); background: rgba(246,195,67,0.10);}
.pill.bad{border-color: rgba(255,93,108,0.55); background: rgba(255,93,108,0.10);}
.hr{height:1px;background:var(--border); margin:10px 0 12px 0;}
.note{color:var(--muted); font-size:0.85rem; line-height:1.35;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("## ONE‑PAGER EJECUTIVO — EVALUACIÓN FINANCIERA")
st.markdown(
    f"<div class='small-muted'>Universidad San Ignacio de Loyola (USIL) — MBA — Proyectos de Inversión / Valuation "
    f"&nbsp;·&nbsp; Moneda: <b>PYG</b> "
    f"&nbsp;·&nbsp; Fecha: <b>{date.today().isoformat()}</b></div>",
    unsafe_allow_html=True
)

# ----------------------------
# Sidebar — Inputs
# ----------------------------
with st.sidebar:
    st.markdown("### 🧩 Identificación")
    project = st.text_input("Proyecto", "Proyecto")
    responsible = st.text_input("Responsable", "Docente: Jorge Rojas")

    st.markdown("---")
    st.markdown("### 0) Inversión inicial")
    capex0 = st.number_input("CAPEX Año 0 (inversión inicial)", value=3_500_000_000.0, step=50_000_000.0, min_value=1.0)

    st.markdown("---")
    st.markdown("### 1) CAPM / WACC (Nominal, D/E contable)")
    rf = st.number_input("Rf (%)", value=4.50, step=0.10) / 100
    erp = st.number_input("ERP (%)", value=5.50, step=0.10) / 100
    crp = st.number_input("CRP (%)", value=2.00, step=0.10) / 100
    beta_u = st.number_input("βU (desapalancada)", value=0.90, step=0.05)
    tax_rate = st.number_input("Impuesto (T) (%)", value=10.0, step=0.5) / 100

    st.markdown("---")
    st.markdown("### 2) Estructura de capital (valores contables)")
    debt = st.number_input("Deuda (D)", value=400_000_000.0, step=10_000_000.0, min_value=0.0)
    equity = st.number_input("Capital propio (E)", value=600_000_000.0, step=10_000_000.0, min_value=1.0)
    kd = st.number_input("Costo de deuda Kd (%)", value=7.00, step=0.10) / 100

    st.markdown("---")
    st.markdown("### 3A) Contable Año 1 (para calcular FCF Año 1)")
    sales = st.number_input("Ventas Año 1", value=8_000_000_000.0, step=50_000_000.0, min_value=0.0)
    var_costs = st.number_input("Costos variables Año 1", value=4_400_000_000.0, step=50_000_000.0, min_value=0.0)
    fixed_costs = st.number_input("Costos fijos Año 1", value=2_000_000_000.0, step=50_000_000.0, min_value=0.0)
    depreciation = st.number_input("Depreciación Año 1 (no caja)", value=350_000_000.0, step=10_000_000.0, min_value=0.0)
    capex_y1 = st.number_input("CAPEX Año 1 (mantenimiento/crecimiento)", value=250_000_000.0, step=10_000_000.0, min_value=0.0)

    st.markdown("**Δ Capital de trabajo (Año 1) — componentes (simplificado)**")
    delta_ar = st.number_input("Δ Cuentas por cobrar (AR)", value=120_000_000.0, step=5_000_000.0)
    delta_inv = st.number_input("Δ Inventarios (INV)", value=80_000_000.0, step=5_000_000.0)
    delta_ap = st.number_input("Δ Cuentas por pagar (AP)", value=60_000_000.0, step=5_000_000.0)

    st.markdown("---")
    st.markdown("### 3B) Flujos (Ciclo 1..N) + g perpetuidad")
    n_years = st.slider("Años de proyección (N)", 3, 10, 5)
    g_exp = st.number_input("g explícito (%) (crecimiento ciclo 1..N)", value=5.0, step=0.25) / 100
    g_inf = st.number_input("g a perpetuidad (%)", value=2.0, step=0.10) / 100

    st.markdown("---")
    st.markdown("### 4) Monte Carlo (incertidumbre razonable)")
    sims = st.slider("Simulaciones", 5_000, 60_000, DEFAULT_SIMS, step=1_000)
    st.caption("Se asume RNG interno (sin semilla expuesta).")

    st.markdown("**Rangos triangulares** (mínimo = adverso, base = más probable, máximo = favorable)")
    g_min = st.number_input("g mín (%)", value=1.0, step=0.25) / 100
    g_mode = st.number_input("g base (%)", value=5.0, step=0.25) / 100
    g_max = st.number_input("g máx (%)", value=9.0, step=0.25) / 100

    w_spread = st.number_input("WACC rango ± (%)", value=2.0, step=0.25) / 100

    capex_min = st.number_input("CAPEX mín (favorable)", value=capex0 * 0.90, step=10_000_000.0)
    capex_mode = st.number_input("CAPEX base", value=capex0, step=10_000_000.0)
    capex_max = st.number_input("CAPEX máx (adverso)", value=capex0 * 1.15, step=10_000_000.0)

    fcf_mult_min = st.number_input("Shock FCF1 mín (adverso)", value=0.85, step=0.01)
    fcf_mult_mode = st.number_input("Shock FCF1 base", value=1.00, step=0.01)
    fcf_mult_max = st.number_input("Shock FCF1 máx (favorable)", value=1.12, step=0.01)

# ----------------------------
# Deterministic calculations
# ----------------------------
beta_l = beta_u * (1 + (1 - tax_rate) * (debt / max(equity, 1e-9)))
ke = rf + beta_l * erp + crp

V = debt + equity
wd = (debt / V) if V > 0 else 0.0
we = (equity / V) if V > 0 else 1.0
wacc = we * ke + wd * kd * (1 - tax_rate)

# Contable Año 1 -> FCF Año 1
ebit = sales - var_costs - fixed_costs - depreciation
nopat = ebit * (1 - tax_rate)
delta_nwc = (delta_ar + delta_inv - delta_ap)
fcf_y1 = nopat + depreciation - capex_y1 - delta_nwc

# Flujos 1..N
years = np.arange(1, n_years + 1)
fcf_explicit = np.array([fcf_y1 * ((1 + g_exp) ** (t - 1)) for t in years], dtype=float)

# Terminal value
tv_ok = wacc > (g_inf + MIN_SPREAD)
tv = (fcf_explicit[-1] * (1 + g_inf) / (wacc - g_inf)) if tv_ok else np.nan

disc = (1 + wacc) ** years
pv_fcf = float(np.sum(fcf_explicit / disc))
pv_tv = float(tv / ((1 + wacc) ** n_years)) if np.isfinite(tv) else np.nan
pv_tv_safe = pv_tv if np.isfinite(pv_tv) else 0.0
npv_base = pv_fcf + pv_tv_safe - capex0

cf_irr = np.concatenate([np.array([-capex0]), fcf_explicit.copy()])
if np.isfinite(tv):
    cf_irr[-1] += tv
irr_base = safe_irr(cf_irr)
pb_simple = payback_simple(cf_irr)
pb_disc = payback_discounted(cf_irr, wacc)

# ----------------------------
# Monte Carlo
# ----------------------------
@st.cache_data(show_spinner=False)
def run_monte_carlo(
    sims: int,
    fcf_y1: float,
    n_years: int,
    g_inf: float,
    min_spread: float,
    g_min: float, g_mode: float, g_max: float,
    w_min: float, w_mode: float, w_max: float,
    capex_min: float, capex_mode: float, capex_max: float,
    m_min: float, m_mode: float, m_max: float,
):
    rng = np.random.default_rng()

    g_s = rng.triangular(g_min, g_mode, g_max, sims)
    w_s = rng.triangular(w_min, w_mode, w_max, sims)
    capex_s = rng.triangular(capex_min, capex_mode, capex_max, sims)
    m_s = rng.triangular(m_min, m_mode, m_max, sims)

    yrs = np.arange(1, n_years + 1)
    fcf1 = fcf_y1 * m_s
    fcf_paths = fcf1[:, None] * (1 + g_s)[:, None] ** (yrs[None, :] - 1)

    valid = w_s > (g_inf + min_spread)
    npv_s = np.full(sims, np.nan)

    idx = np.where(valid)[0]
    if idx.size == 0:
        return npv_s, valid, g_s, w_s, capex_s, m_s

    fcf_v = fcf_paths[idx, :].copy()
    w_v = w_s[idx]
    cap_v = capex_s[idx]

    tv_v = (fcf_v[:, -1] * (1 + g_inf)) / (w_v - g_inf)
    fcf_v[:, -1] += tv_v

    disc_v = (1 + w_v)[:, None] ** (yrs[None, :])
    pv = np.sum(fcf_v / disc_v, axis=1)
    npv_s[idx] = pv - cap_v

    return npv_s, valid, g_s, w_s, capex_s, m_s

w_min = max(0.0001, wacc - w_spread)
w_mode = max(0.0001, wacc)
w_max = max(w_min + 0.0001, wacc + w_spread)

npv_sims, valid_mask, g_s, w_s, cap_s, m_s = run_monte_carlo(
    sims=sims,
    fcf_y1=fcf_y1,
    n_years=n_years,
    g_inf=g_inf,
    min_spread=MIN_SPREAD,
    g_min=g_min, g_mode=g_mode, g_max=g_max,
    w_min=w_min, w_mode=w_mode, w_max=w_max,
    capex_min=capex_min, capex_mode=capex_mode, capex_max=capex_max,
    m_min=fcf_mult_min, m_mode=fcf_mult_mode, m_max=fcf_mult_max
)

valid_npvs = npv_sims[np.isfinite(npv_sims)]
valid_rate = float(np.mean(valid_mask)) if valid_mask.size else 0.0

if valid_npvs.size == 0:
    prob_neg = np.nan
    p5 = p50 = p95 = np.nan
    mean_npv = std_npv = cvar5 = np.nan
else:
    prob_neg = float(np.mean(valid_npvs < 0))
    p5, p50, p95 = np.percentile(valid_npvs, [5, 50, 95])
    mean_npv = float(np.mean(valid_npvs))
    std_npv = float(np.std(valid_npvs))
    cvar5 = cvar_left_tail(valid_npvs, 0.05)

verdict, rationale, checks = committee_verdict(
    npv_base=npv_base,
    prob_neg=float(prob_neg) if np.isfinite(prob_neg) else 1.0,
    p5=float(p5) if np.isfinite(p5) else -1e18,
    p50=float(p50) if np.isfinite(p50) else -1e18
)

def pill_class(v):
    return "good" if v == "APROBADO" else ("warn" if v == "OBSERVADO" else "bad")

# ----------------------------
# KPI row
# ----------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
with k1:
    st.markdown(f"<div class='kpi'><div class='label'>VAN (base)</div><div class='value'>{fmt_gs(npv_base)}</div><div class='sub'>Determinístico</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi'><div class='label'>TIR (base)</div><div class='value'>{fmt_pct(irr_base)}</div><div class='sub'>Determinístico</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi'><div class='label'>Payback</div><div class='value'>{(f'{pb_simple:.2f} años') if pb_simple is not None else 'N/A'}</div><div class='sub'>Simple</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi'><div class='label'>Payback (desc.)</div><div class='value'>{(f'{pb_disc:.2f} años') if pb_disc is not None else 'N/A'}</div><div class='sub'>Descontado</div></div>", unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='kpi'><div class='label'>P(VAN&lt;0)</div><div class='value'>{(f'{prob_neg*100:.1f}%') if np.isfinite(prob_neg) else 'N/A'}</div><div class='sub'>Monte Carlo</div></div>", unsafe_allow_html=True)
with k6:
    st.markdown(f"<div class='kpi'><div class='label'>P50 (VAN)</div><div class='value'>{fmt_gs(p50) if np.isfinite(p50) else 'N/A'}</div><div class='sub'>Monte Carlo</div></div>", unsafe_allow_html=True)

# ----------------------------
# Middle row
# ----------------------------
c_left, c_right = st.columns([1.10, 0.90], gap="large")

with c_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Decisión & supuestos")
    st.markdown(f"<span class='pill {pill_class(verdict)}'>DICTAMEN: {verdict}</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='note'>{rationale}</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    colA, colB = st.columns([1, 1], gap="small")
    with colA:
        st.markdown("**Supuestos clave**")
        st.markdown(
            f"- Horizonte: **{n_years} años** + perpetuidad\n"
            f"- g explícito: **{fmt_pct(g_exp)}**  |  g∞: **{fmt_pct(g_inf)}**\n"
            f"- WACC: **{fmt_pct(wacc)}**  (Ke {fmt_pct(ke)} | Kd {fmt_pct(kd)})\n"
            f"- CAPEX₀: **{fmt_gs(capex0)}**\n"
            f"- FCF Año 1 (calculado): **{fmt_gs(fcf_y1)}**"
        )
    with colB:
        st.markdown("**Checklist Comité (automático)**")
        for k, v in checks.items():
            st.markdown(f"- {'✅' if v else '❌'} {k}")
        st.markdown("")
        st.markdown("**Puente contable → FCF (Año 1)**")
        st.markdown(
            f"- EBIT: {fmt_gs(ebit)}\n"
            f"- NOPAT: {fmt_gs(nopat)}\n"
            f"- ΔCT (AR+INV−AP): {fmt_gs(delta_nwc)}\n"
            f"- CAPEX Año 1: {fmt_gs(capex_y1)}"
        )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    exec_read = (
        f"Con {n_years} años explícitos y g∞={fmt_pct(g_inf)}, el VAN base es {fmt_gs(npv_base)} "
        f"y la TIR base {fmt_pct(irr_base) if irr_base is not None else 'N/A'}. "
        f"En Monte Carlo, P(VAN<0)={prob_neg*100:.1f}% y P5={fmt_gs(p5)} "
        f"(escenario adverso plausible). Conclusión: {verdict}."
    )
    st.markdown("**Lectura ejecutiva (1 párrafo)**")
    st.markdown(f"<div class='note'>{exec_read}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Estructura del valor")
    st.markdown("<div class='note'>VAN = PV(FCF 1..N) + PV(TV) − CAPEX₀. Se reporta TV separado para transparencia.</div>", unsafe_allow_html=True)

    df = pd.DataFrame({
        "Componente": ["PV FCF", "PV TV", "- CAPEX₀"],
        "Valor": [pv_fcf, pv_tv_safe, -capex0]
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Componente"], y=df["Valor"],
        text=[fmt_gs(v) for v in df["Valor"]],
        textposition="outside",
        hovertemplate="%{x}<br>%{text}<extra></extra>"
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="", showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
        xaxis=dict(title=""),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**VAN (base):** <span class='pill good'>{fmt_gs(npv_base)}</span>", unsafe_allow_html=True)
    if not tv_ok:
        st.warning("Consistencia TV: requiere WACC > g∞ + 0.5%. Ajustá supuestos.", icon="⚠️")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Bottom row
# ----------------------------
b1, b2 = st.columns(2, gap="large")

with b1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Flujos proyectados (ciclo 1..N) + TV")
    st.markdown("<div class='note'>FCF por ciclo. El valor terminal (TV) se muestra separado.</div>", unsafe_allow_html=True)

    df_fcf = pd.DataFrame({"Año": years, "FCF": fcf_explicit})
    fig_fcf = px.bar(df_fcf, x="Año", y="FCF")
    fig_fcf.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="", showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
        xaxis=dict(title="Año"),
    )
    st.plotly_chart(fig_fcf, use_container_width=True)
    st.markdown(f"<div class='note'>TV (en ciclo {n_years}): <b>{fmt_gs(tv) if np.isfinite(tv) else 'N/A'}</b></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Riesgo (Monte Carlo) — interpretación ejecutiva")
    if valid_npvs.size == 0:
        st.error("No hay simulaciones válidas (WACC ≤ g∞ + spread). Ajustá supuestos.")
    else:
        st.markdown(
            f"<div class='note'>"
            f"**Cobertura:** {valid_rate*100:.1f}% simulaciones válidas (TV consistente). "
            f"**P(VAN&lt;0):** {prob_neg*100:.1f}%. "
            f"**P5/P50/P95:** {fmt_gs(p5)} / {fmt_gs(p50)} / {fmt_gs(p95)}. "
            f"**Media:** {fmt_gs(mean_npv)}. "
            f"**Volatilidad (σ):** {fmt_gs(std_npv)}. "
            f"**CVaR5:** {fmt_gs(cvar5)} (promedio del 5% peor)."
            f"</div>",
            unsafe_allow_html=True
        )

        # Histograma interactivo
        df_hist = pd.DataFrame({"VAN": valid_npvs})
        fig_hist = px.histogram(df_hist, x="VAN", nbins=HIST_BINS)
        fig_hist.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Frecuencia", showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
            xaxis=dict(title="VAN (PYG)"),
            showlegend=False
        )
        for val, name in [(p5, "P5"), (p50, "P50"), (p95, "P95")]:
            fig_hist.add_vline(x=val, line_width=2, line_dash="dash", annotation_text=name, annotation_position="top")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Texto ejecutivo: qué significa cada percentil
        expl = (
            "• **P50** es el escenario “central”: 50% de los resultados quedan por encima y 50% por debajo.\n"
            "• **P5** representa un escenario adverso plausible: solo 5% de los casos serían peores.\n"
            "• **P95** representa un escenario favorable plausible: solo 5% de los casos serían mejores.\n"
            "• **CVaR5** resume el “daño esperado” en la cola izquierda (los peores 5%)."
        )
        st.markdown(expl)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='note'>Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence.</div>", unsafe_allow_html=True)

# ----------------------------
# Export — One‑Pager TXT / PDF
# ----------------------------
def build_onepager_text() -> str:
    lines = []
    lines.append("ONE‑PAGER EJECUTIVO — EVALUACIÓN FINANCIERA (PYG)")
    lines.append(f"Fecha: {date.today().isoformat()}")
    lines.append(f"Proyecto: {project}")
    lines.append(f"Responsable: {responsible}")
    lines.append("")
    lines.append(f"DICTAMEN: {verdict}")
    lines.append(rationale)
    lines.append("")
    lines.append("KPIs (determinístico)")
    lines.append(f"- VAN: {fmt_gs(npv_base)}")
    lines.append(f"- TIR: {fmt_pct(irr_base)}")
    lines.append(f"- Payback: {f'{pb_simple:.2f} años' if pb_simple is not None else 'N/A'}")
    lines.append(f"- Payback (desc.): {f'{pb_disc:.2f} años' if pb_disc is not None else 'N/A'}")
    lines.append("")
    lines.append("Flujos proyectados (FCF) — ciclo 1..N")
    for y, v in zip(years, fcf_explicit):
        lines.append(f"- Año {int(y)}: {fmt_gs(float(v))}")
    lines.append(f"Valor Terminal (TV, en ciclo {n_years}): {fmt_gs(tv) if np.isfinite(tv) else 'N/A'}")
    lines.append("")
    lines.append("Riesgo (Monte Carlo)")
    if np.isfinite(prob_neg):
        lines.append(f"- Simulaciones: {sims:,} | válidas: {valid_rate*100:.1f}%")
        lines.append(f"- P(VAN<0): {prob_neg*100:.1f}%")
        lines.append(f"- P5: {fmt_gs(p5)} | P50: {fmt_gs(p50)} | P95: {fmt_gs(p95)}")
        lines.append(f"- Media: {fmt_gs(mean_npv)} | σ: {fmt_gs(std_npv)} | CVaR5: {fmt_gs(cvar5)}")
    else:
        lines.append("- No disponible (sin simulaciones válidas).")
    lines.append("")
    lines.append("Nota: uso académico. No sustituye due diligence.")
    return "\n".join(lines)

def pdf_draw_histogram(c: canvas.Canvas, x, y, w, h, values: np.ndarray, bins: int = 30):
    if values.size == 0:
        return
    counts, edges = np.histogram(values, bins=bins)
    m = counts.max() if counts.max() > 0 else 1
    bar_w = w / bins
    c.setStrokeColor(HexColor("#1c2a45"))
    c.setFillColor(HexColor("#4aa3ff"))
    for i in range(bins):
        bh = (counts[i] / m) * h
        c.rect(x + i * bar_w, y, bar_w * 0.92, bh, stroke=0, fill=1)
    c.setFillColor(HexColor("#a7b4d6"))
    c.setFont("Helvetica", 7)
    c.drawString(x, y - 10, "Distribución del VAN (Monte Carlo)")

def generate_pdf_onepager() -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab no está disponible. Agregá `reportlab` a requirements.txt.")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(letter))
    W, H = landscape(letter)

    bg = HexColor("#070b14")
    card = HexColor("#0f1628")
    border = HexColor("#1c2a45")
    txt = HexColor("#e9eefc")
    muted = HexColor("#a7b4d6")
    good = HexColor("#29d07f")
    warn = HexColor("#f6c343")
    bad = HexColor("#ff5d6c")

    def pill_color(v):
        return good if v == "APROBADO" else (warn if v == "OBSERVADO" else bad)

    c.setFillColor(bg); c.rect(0, 0, W, H, stroke=0, fill=1)

    margin = 0.55 * inch
    gx = margin
    gy = margin
    gw = W - 2*margin

    # Header
    c.setFillColor(txt); c.setFont("Helvetica-Bold", 14)
    c.drawString(gx, H - margin + 10, "ONE‑PAGER EJECUTIVO — EVALUACIÓN FINANCIERA (PYG)")
    c.setFillColor(muted); c.setFont("Helvetica", 9)
    c.drawRightString(W - margin, H - margin + 12, f"Fecha: {date.today().isoformat()}")
    c.drawString(gx, H - margin - 4, "USIL — MBA — Proyectos de Inversión / Valuation")

    # Subheader
    c.setFillColor(muted); c.setFont("Helvetica", 9)
    c.drawString(gx, H - margin - 20, f"Proyecto: {project}  |  Responsable: {responsible}")

    # Decision pill
    c.setFillColor(pill_color(verdict))
    c.roundRect(gx, H - margin - 46, 135, 18, 9, stroke=0, fill=1)
    c.setFillColor(HexColor("#071014")); c.setFont("Helvetica-Bold", 8)
    c.drawString(gx + 10, H - margin - 42, f"DICTAMEN: {verdict}")

    # KPI strip
    k_y = H - margin - 85
    k_h = 46
    k_gap = 10
    k_w = (gw - 5*k_gap) / 6

    def kpi(i, label, value, sub):
        x = gx + i * (k_w + k_gap)
        c.setFillColor(card); c.setStrokeColor(border); c.roundRect(x, k_y, k_w, k_h, 10, stroke=1, fill=1)
        c.setFillColor(muted); c.setFont("Helvetica", 7.5); c.drawString(x+10, k_y+k_h-16, label)
        c.setFillColor(txt); c.setFont("Helvetica-Bold", 10); c.drawString(x+10, k_y+k_h-32, value)
        c.setFillColor(muted); c.setFont("Helvetica", 7); c.drawString(x+10, k_y+10, sub)

    kpi(0, "VAN (base)", fmt_gs(npv_base), "Determinístico")
    kpi(1, "TIR (base)", fmt_pct(irr_base), "Determinístico")
    kpi(2, "Payback", f"{pb_simple:.2f} a" if pb_simple is not None else "N/A", "Simple")
    kpi(3, "Payback (desc.)", f"{pb_disc:.2f} a" if pb_disc is not None else "N/A", "Descontado")
    kpi(4, "P(VAN<0)", f"{prob_neg*100:.1f}%" if np.isfinite(prob_neg) else "N/A", "Monte Carlo")
    kpi(5, "P50 (VAN)", fmt_gs(p50) if np.isfinite(p50) else "N/A", "Monte Carlo")

    # Cards area
    top = k_y - 16
    card_h = 150
    gap = 14
    col_w = (gw - gap) / 2

    def card_box(x, y, w, h, title):
        c.setFillColor(card); c.setStrokeColor(border); c.roundRect(x, y, w, h, 14, stroke=1, fill=1)
        c.setFillColor(txt); c.setFont("Helvetica-Bold", 10); c.drawString(x+12, y+h-22, title)
        c.setStrokeColor(border); c.setLineWidth(1); c.line(x+12, y+h-28, x+w-12, y+h-28)

    # Left card: Executive summary
    x1 = gx
    y1 = top - card_h
    card_box(x1, y1, col_w, card_h, "Resumen ejecutivo")
    c.setFillColor(muted); c.setFont("Helvetica", 8)
    txt_lines = textwrap.wrap(rationale, width=70)
    yy = y1 + card_h - 46
    for ln in txt_lines[:4]:
        c.drawString(x1+12, yy, ln); yy -= 12

    c.setFillColor(muted); c.setFont("Helvetica", 7.8)
    bullets = [
        f"Horizonte: {n_years} años + perpetuidad",
        f"WACC: {fmt_pct(wacc)} (Ke {fmt_pct(ke)} | Kd {fmt_pct(kd)})",
        f"g explícito: {fmt_pct(g_exp)} | g∞: {fmt_pct(g_inf)}",
        f"CAPEX₀: {fmt_gs(capex0)} | FCF1: {fmt_gs(fcf_y1)}",
    ]
    yy -= 2
    for b in bullets:
        c.drawString(x1+12, yy, "• " + b); yy -= 11

    # Right card: Monte Carlo distribution
    x2 = gx + col_w + gap
    y2 = y1
    card_box(x2, y2, col_w, card_h, "Monte Carlo — distribución del VAN")
    c.setFillColor(muted); c.setFont("Helvetica", 8)
    if valid_npvs.size > 0:
        c.drawString(x2+12, y2+card_h-46, f"Simulaciones: {sims:,} | válidas: {valid_rate*100:.1f}% | P(VAN<0): {prob_neg*100:.1f}%")
        c.drawString(x2+12, y2+card_h-60, f"P5/P50/P95: {fmt_gs(p5)} / {fmt_gs(p50)} / {fmt_gs(p95)}")
        c.drawString(x2+12, y2+card_h-74, f"Media: {fmt_gs(mean_npv)} | σ: {fmt_gs(std_npv)} | CVaR5: {fmt_gs(cvar5)}")
        pdf_draw_histogram(c, x2+16, y2+22, col_w-32, 70, valid_npvs, bins=30)
    else:
        c.drawString(x2+12, y2+card_h-46, "Sin simulaciones válidas (revisar WACC vs g∞).")

    # Footer
    c.setFillColor(muted); c.setFont("Helvetica", 7)
    c.drawString(gx, gy-6, "Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence.")
    c.drawRightString(W - margin, gy-6, "ValuationSuite USIL (PYG)")

    c.showPage(); c.save()
    return buf.getvalue()

# Export section
st.markdown("### Export")
c1, c2 = st.columns([1, 1], gap="small")
with c1:
    txt = build_onepager_text()
    st.download_button("⬇️ Descargar One‑Pager (TXT)", data=txt.encode("utf-8"), file_name="one_pager_usil_PYG.txt")
with c2:
    if REPORTLAB_OK:
        try:
            pdf_bytes = generate_pdf_onepager()
            st.download_button("⬇️ Descargar One‑Pager (PDF)", data=pdf_bytes, file_name="one_pager_usil_PYG.pdf", mime="application/pdf")
            st.caption("PDF 1 página (sin matplotlib).")
        except Exception as e:
            st.warning(f"No se pudo generar PDF: {e}")
    else:
        st.info("Para exportar PDF, agrega `reportlab` a requirements.txt.")
