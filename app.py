# app.py
# ValuationSuite USIL (PYG) — DCF + Monte Carlo + One-Pager Ejecutivo (Dashboard + PDF)
# Autor: (generado con ChatGPT a pedido del docente)
#
# ✅ Ajustes implementados:
# - Moneda fija: PYG
# - Sin selectores "Base de medición" ni "Base D/E": se asume Nominal y valores contables
# - Sin selector de semilla en Monte Carlo (usa RNG interno)
# - Criterios de comité fijos por default: P50>0, P5>0, P(VAN<0) <= 20% y VAN base > 0
# - Inputs ampliados:
#   * Hoja/Sección "Contable Año 1": ventas, costos variables/fijos, depreciación, CAPEX, ΔCT (componentes)
#   * Se calculan Utilidad, NOPAT y FCF Año 1 automáticamente
#   * Hoja/Sección "Flujos": se muestra explícitamente FCF ciclo 1..N y g perpetuidad
# - KPIs: VAN, TIR, Payback simple y descontado (y TV separado)
# - One-pager ejecutivo: dashboard premium + export PDF 1 página (sin matplotlib)

import io
import textwrap
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ReportLab (para PDF 1 página)
REPORTLAB_OK = True
try:
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
except Exception:
    REPORTLAB_OK = False

# ----------------------------
# Config & guardrails
# ----------------------------
st.set_page_config(page_title="ValuationSuite USIL — One‑Pager (PYG)", layout="wide")

CURRENCY = "Gs."
MIN_SPREAD = 0.005  # WACC debe ser > g∞ + 0.5%
DEFAULT_SIMS = 15000
MAX_PROB_NEG = 0.20  # Comité: P(VAN<0) <= 20%

# ----------------------------
# Utils
# ----------------------------
def fmt_gs(x: float) -> str:
    try:
        return f"{CURRENCY} {x:,.0f}".replace(",", ".")
    except Exception:
        return f"{CURRENCY} {x}"

def fmt_pct(x: float) -> str:
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "—"

def safe_float(x, default=0.0):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

def safe_irr(cashflows: np.ndarray):
    # npf.irr puede devolver NaN o valores extremos en casos numéricos raros
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
    # cashflows: array con CF0 negativo, CF1.. positivo
    cum = 0.0
    for t in range(len(cashflows)):
        cum += cashflows[t]
        if cum >= 0 and t > 0:
            prev_cum = cum - cashflows[t]
            # interpolación lineal dentro del año t
            if cashflows[t] == 0:
                return float(t)
            frac = (0 - prev_cum) / cashflows[t]
            return float(t - 1 + frac)
    return None

def payback_discounted(cashflows: np.ndarray, wacc: float):
    cum = 0.0
    for t in range(len(cashflows)):
        pv = cashflows[t] / ((1 + wacc) ** t)
        cum += pv
        if cum >= 0 and t > 0:
            prev_cum = cum - pv
            if pv == 0:
                return float(t)
            frac = (0 - prev_cum) / pv
            return float(t - 1 + frac)
    return None

def committee_verdict(npv_base, prob_neg, p5, p50):
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
        return "RECHAZADO", "El proyecto no cumple criterios mínimos del comité. Requiere rediseño de supuestos y mitigaciones.", checks
    return "OBSERVADO", "El proyecto muestra potencial, pero requiere evidencia/mitigaciones en supuestos críticos antes de aprobación.", checks

def wrap_lines(text: str, width=90):
    return textwrap.fill(text, width=width)

# ----------------------------
# Styling (premium dashboard)
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
.block-container {padding-top: 1.0rem; padding-bottom: 1.5rem; max-width: 1200px;}
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
  height: 94px;
}
.kpi .label{color:var(--muted); font-size:0.85rem; margin-bottom:6px;}
.kpi .value{color:var(--txt); font-size:1.25rem; font-weight:700; line-height:1.1;}
.kpi .sub{color:var(--muted); font-size:0.80rem; margin-top:6px;}
.pill{
  display:inline-block;
  padding:4px 10px;
  border-radius:999px;
  font-size:0.82rem;
  border:1px solid var(--border);
  color:var(--txt);
  background: rgba(255,255,255,0.03);
}
.pill.good{border-color: rgba(41,208,127,0.55); background: rgba(41,208,127,0.10);}
.pill.warn{border-color: rgba(246,195,67,0.55); background: rgba(246,195,67,0.10);}
.pill.bad{border-color: rgba(255,93,108,0.55); background: rgba(255,93,108,0.10);}
.hr{height:1px;background:var(--border); margin:10px 0 12px 0;}
.note{color:var(--muted); font-size:0.82rem;}

.headerbar{display:flex;justify-content:space-between;align-items:flex-end;gap:14px;flex-wrap:wrap}
.hb-title{font-size:18px;font-weight:800;letter-spacing:.2px;line-height:1.2;margin-bottom:4px}
.hb-sub{font-size:12px;opacity:.85;line-height:1.35}
.hb-right{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;justify-content:flex-end}
.pill{padding:6px 10px;border:1px solid rgba(255,255,255,.12);border-radius:999px;background:rgba(255,255,255,.04);font-size:12px}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
today_str = date.today().isoformat()

st.markdown(
    f"""
    <div class="card headerbar">
      <div class="hb-left">
        <div class="hb-title">ONE-PAGER EJECUTIVO — EVALUACIÓN FINANCIERA</div>
        <div class="hb-sub">
          Universidad San Ignacio de Loyola (USIL) — Maestría en Administración de Negocios (MBA) — Proyectos de Inversión / Valuation
        </div>
      </div>
      <div class="hb-right">
        <div class="pill">Moneda: <b>PYG</b></div>
        <div class="pill">Fecha: <b>{today_str}</b></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
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

    # WACC alrededor del calculado (±2%)
    w_spread = st.number_input("WACC rango ± (%)", value=2.0, step=0.25) / 100

    capex_min = st.number_input("CAPEX mín (favorable)", value=capex0 * 0.90, step=10_000_000.0)
    capex_mode = st.number_input("CAPEX base", value=capex0, step=10_000_000.0)
    capex_max = st.number_input("CAPEX máx (adverso)", value=capex0 * 1.15, step=10_000_000.0)

    fcf_mult_min = st.number_input("Shock FCF1 mín (adverso)", value=0.85, step=0.01)
    fcf_mult_mode = st.number_input("Shock FCF1 base", value=1.00, step=0.01)
    fcf_mult_max = st.number_input("Shock FCF1 máx (favorable)", value=1.12, step=0.01)

# ----------------------------
# Cálculos determinísticos
# ----------------------------
# CAPM + WACC
beta_l = beta_u * (1 + (1 - tax_rate) * (debt / max(equity, 1e-9)))
ke = rf + beta_l * erp + crp

V = debt + equity
wd = (debt / V) if V > 0 else 0.0
we = (equity / V) if V > 0 else 1.0
wacc = we * ke + wd * kd * (1 - tax_rate)

# Contable Año 1 -> FCF Año 1
ebit = sales - var_costs - fixed_costs - depreciation
nopat = ebit * (1 - tax_rate)
delta_nwc = (delta_ar + delta_inv - delta_ap)  # aumento de NWC => consumo de caja
fcf_y1 = nopat + depreciation - capex_y1 - delta_nwc

# Flujos explícitos ciclo 1..N (desde FCF año 1)
years = np.arange(1, n_years + 1)
fcf_explicit = np.array([fcf_y1 * ((1 + g_exp) ** (t - 1)) for t in years], dtype=float)

# Valor terminal (Gordon)
tv_ok = wacc > (g_inf + MIN_SPREAD)
tv = (fcf_explicit[-1] * (1 + g_inf) / (wacc - g_inf)) if tv_ok else np.nan

# PV de flujos y TV
disc = (1 + wacc) ** years
pv_fcf = float(np.sum(fcf_explicit / disc))
pv_tv = float(tv / ((1 + wacc) ** n_years)) if np.isfinite(tv) else np.nan
npv_base = pv_fcf + (pv_tv if np.isfinite(pv_tv) else 0.0) - capex0

# TIR base (incluye TV como parte del último flujo)
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
    # g triangular
    g_min: float, g_mode: float, g_max: float,
    # wacc triangular (derivado)
    w_min: float, w_mode: float, w_max: float,
    # capex triangular
    capex_min: float, capex_mode: float, capex_max: float,
    # shock FCF1 triangular
    m_min: float, m_mode: float, m_max: float,
):
    rng = np.random.default_rng()

    g_s = rng.triangular(g_min, g_mode, g_max, sims)
    w_s = rng.triangular(w_min, w_mode, w_max, sims)
    capex_s = rng.triangular(capex_min, capex_mode, capex_max, sims)
    m_s = rng.triangular(m_min, m_mode, m_max, sims)

    yrs = np.arange(1, n_years + 1)
    fcf1 = fcf_y1 * m_s
    fcf_paths = fcf1[:, None] * (1.0 + g_s)[:, None] ** (yrs[None, :] - 1)

    valid = w_s > (g_inf + min_spread)
    npv_s = np.full(sims, np.nan)

    idx = np.where(valid)[0]
    if idx.size == 0:
        return npv_s, g_s, w_s, capex_s, m_s, idx

    fcf_v = fcf_paths[idx, :].copy()
    w_v = w_s[idx]
    cap_v = capex_s[idx]

    tv_v = (fcf_v[:, -1] * (1 + g_inf)) / (w_v - g_inf)
    fcf_v[:, -1] += tv_v

    disc_v = (1 + w_v)[:, None] ** (yrs[None, :])
    pv = np.sum(fcf_v / disc_v, axis=1)
    npv_s[idx] = pv - cap_v
    return npv_s, g_s, w_s, capex_s, m_s, idx

w_min = max(0.0001, wacc - w_spread)
w_mode = max(0.0001, wacc)
w_max = max(w_min + 0.0001, wacc + w_spread)

npv_sims, g_s, w_s, cap_s, m_s, valid_idx = run_monte_carlo(
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
if valid_npvs.size == 0:
    prob_neg = np.nan
    p5 = p50 = p95 = np.nan
    mean_npv = std_npv = skew_npv = cvar5 = valid_rate = np.nan
else:
    prob_neg = float(np.mean(valid_npvs < 0))
    p5, p50, p95 = np.percentile(valid_npvs, [5, 50, 95])
    mean_npv = float(np.mean(valid_npvs))
    std_npv = float(np.std(valid_npvs, ddof=1)) if valid_npvs.size > 1 else 0.0
    skew_npv = float(np.mean(((valid_npvs-mean_npv)/(std_npv if std_npv>0 else 1.0))**3)) if valid_npvs.size > 2 else 0.0
    cvar5 = float(np.mean(valid_npvs[valid_npvs <= p5])) if valid_npvs.size > 0 else np.nan
    valid_rate = float(np.mean(np.isfinite(npv_sims)))

verdict, rationale, checks = committee_verdict(npv_base=npv_base, prob_neg=prob_neg if np.isfinite(prob_neg) else 1.0, p5=p5 if np.isfinite(p5) else -1e18, p50=p50 if np.isfinite(p50) else -1e18)

def pill_class(v):
    return "good" if v=="APROBADO" else ("warn" if v=="OBSERVADO" else "bad")

# ----------------------------
# KPI row
# ----------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")

with k1:
    st.markdown(f"<div class='kpi'><div class='label'>VAN (base)</div><div class='value'>{fmt_gs(npv_base)}</div><div class='sub'>Determinístico</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi'><div class='label'>TIR (base)</div><div class='value'>{fmt_pct(irr_base) if irr_base is not None else 'N/A'}</div><div class='sub'>Determinístico</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi'><div class='label'>Payback</div><div class='value'>{(f'{pb_simple:.2f} años') if pb_simple is not None else 'N/A'}</div><div class='sub'>Simple</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi'><div class='label'>Payback (desc.)</div><div class='value'>{(f'{pb_disc:.2f} años') if pb_disc is not None else 'N/A'}</div><div class='sub'>Descontado</div></div>", unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='kpi'><div class='label'>P(VAN&lt;0)</div><div class='value'>{(f'{prob_neg*100:.1f}%') if np.isfinite(prob_neg) else 'N/A'}</div><div class='sub'>Monte Carlo</div></div>", unsafe_allow_html=True)
with k6:
    st.markdown(f"<div class='kpi'><div class='label'>P50 (VAN)</div><div class='value'>{fmt_gs(p50) if np.isfinite(p50) else 'N/A'}</div><div class='sub'>Monte Carlo</div></div>", unsafe_allow_html=True)

# ----------------------------
# Middle row (2 cards) — acá estaba “corto”: lo hacemos más lleno y balanceado
# ----------------------------
c_left, c_right = st.columns([1.08, 0.92], gap="large")

with c_left:
    # Card: decisión + supuestos + checklist
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Decisión & supuestos")
    st.markdown(
        f"<span class='pill {pill_class(verdict)}'>DICTAMEN: {verdict}</span>",
        unsafe_allow_html=True
    )
    st.markdown(f"<div class='note'>{rationale}</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    colA, colB = st.columns([1,1], gap="small")
    with colA:
        st.markdown("**Supuestos clave**")
        st.markdown(
            f"- Horizonte: **{n_years} años** + perpetuidad\n"
            f"- g explícito: **{fmt_pct(g_exp)}**  |  g∞: **{fmt_pct(g_inf)}**\n"
            f"- WACC: **{fmt_pct(wacc)}**  (Ke {fmt_pct(ke)} | Kd {fmt_pct(kd)})\n"
            f"- CAPEX 0: **{fmt_gs(capex0)}**\n"
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
    st.markdown("**Lectura ejecutiva (1 párrafo)**")
    executive_read = (
        f"Con {n_years} años explícitos y crecimiento a perpetuidad g∞={fmt_pct(g_inf)}, el VAN base es {fmt_gs(npv_base)} "
        f"y la TIR base {fmt_pct(irr_base) if irr_base is not None else 'N/A'}. "
        f"El downside bajo Monte Carlo se resume en P(VAN<0)={prob_neg*100:.1f}% y P5={fmt_gs(p5)}. "
        f"Conclusión: {verdict}."
    )
    st.markdown(f"<div class='note'>{executive_read}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c_right:
    # Card: estructura del valor (PV FCF, PV TV, CAPEX) y mini “waterfall”
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Estructura del valor")
    st.markdown("<div class='note'>VAN = PV(FCF ciclo 1..N) + PV(TV) − CAPEX₀. TV se reporta separado para transparencia.</div>", unsafe_allow_html=True)

    pv_tv_safe = pv_tv if np.isfinite(pv_tv) else 0.0
    data = pd.DataFrame({
        "Componente": ["PV FCF", "PV TV", "- CAPEX₀"],
        "Valor": [pv_fcf, pv_tv_safe, -capex0]
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data["Componente"],
        y=data["Valor"],
        text=[fmt_gs(v) for v in data["Valor"]],
        textposition="outside",
        hovertemplate="%{x}<br>%{text}<extra></extra>",
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
        st.warning("Inconsistencia TV: WACC debe ser mayor que g∞ + 0.5%. Ajusta supuestos.", icon="⚠️")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Bottom row (2 charts)
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
    st.markdown("### Riesgo (Monte Carlo) — lectura ejecutiva")
    if valid_npvs.size == 0:
        st.error("No hay simulaciones válidas (WACC ≤ g∞ + spread). Ajusta supuestos.")
    else:
        st.markdown(
            f"<div class='note'>P5: <b>{fmt_gs(p5)}</b> · P50: <b>{fmt_gs(p50)}</b> · P95: <b>{fmt_gs(p95)}</b> · "
            f"P(VAN&lt;0): <b>{prob_neg*100:.1f}%</b> · Simulaciones: <b>{sims:,}</b></div>",
            unsafe_allow_html=True
        )
        df_hist = pd.DataFrame({"VAN": valid_npvs})
        fig_hist = px.histogram(df_hist, x="VAN", nbins=40)
        fig_hist.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="count", showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
            xaxis=dict(title="VAN"),
            showlegend=False
        )
        # Líneas P5, P50, P95
        for val, name in [(p5, "P5"), (p50, "P50"), (p95, "P95")]:
            fig_hist.add_vline(x=val, line_width=2, line_dash="dash", annotation_text=name, annotation_position="top")
        st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("<div class='note'>Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence. · ValuationSuite USIL (PYG)</div>", unsafe_allow_html=True)

# ----------------------------
# Export: One‑Pager PDF (1 página, sin matplotlib)
# ----------------------------
@dataclass
class OnePagerData:
    project: str
    responsible: str

    # Determinístico
    capex0: float
    wacc: float
    ke: float
    kd: float
    g_exp: float
    g_inf: float
    n_years: int
    fcf_y1: float
    fcf_explicit: np.ndarray
    tv: float
    pv_fcf: float
    pv_tv: float
    npv_base: float
    irr_base: float | None
    pb_simple: float | None
    pb_disc: float | None

    # Monte Carlo
    sims: int
    valid_rate: float
    prob_neg: float
    p5: float
    p50: float
    p95: float
    mean_npv: float
    std_npv: float
    skew_npv: float
    cvar5: float
    hist_edges: np.ndarray
    hist_counts: np.ndarray

    # Comité
    verdict: str
    rationale: str

def draw_bar_chart_pdf(c: canvas.Canvas, x, y, w, h, values, labels, color="#4aa3ff"):
    if len(values) == 0:
        return
    vmax = max(values) if max(values) != 0 else 1
    bar_w = w / (len(values) * 1.2)
    gap = bar_w * 0.2
    for i, v in enumerate(values):
        bh = (v / vmax) * h
        bx = x + i * (bar_w + gap)
        by = y
        c.setFillColor(HexColor(color))
        c.rect(bx, by, bar_w, bh, stroke=0, fill=1)
        c.setFillColor(HexColor("#a7b4d6"))
        c.setFont("Helvetica", 7)
        c.drawCentredString(bx + bar_w/2, y - 10, str(labels[i]))

def draw_histogram_pdf(c, x, y, w, h, edges, counts, vlines=None):
    """Histograma minimalista para el One‑Pager PDF (sin matplotlib)."""
    if edges is None or counts is None:
        return
    edges = np.asarray(edges, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0:
        return

    bins = int(counts.size)
    max_c = float(np.max(counts)) if np.max(counts) > 0 else 1.0

    # marco
    c.setStrokeColorRGB(0.35, 0.40, 0.50)
    c.rect(x, y, w, h, stroke=1, fill=0)

    bw = w / bins
    c.setFillColorRGB(0.43, 0.74, 1.00)
    for i in range(bins):
        bh = (float(counts[i]) / max_c) * (h - 6)
        c.rect(x + i*bw, y, bw*0.92, bh, stroke=0, fill=1)

    if vlines:
        xmin = float(edges[0]); xmax = float(edges[-1])
        span = (xmax - xmin) if (xmax - xmin) != 0 else 1.0
        for val, rgb, dashed in vlines:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            t = (float(val) - xmin) / span
            t = max(0.0, min(1.0, t))
            xx = x + t*w
            if rgb is None:
                rgb = (1.0, 1.0, 1.0)
            c.setStrokeColorRGB(*rgb)
            if dashed:
                c.setDash(4, 3)
            else:
                c.setDash()
            c.line(xx, y, xx, y + h)
        c.setDash()


def generate_onepager_pdf(d: OnePagerData) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab no está disponible. Agrega `reportlab` a requirements.txt.")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(letter))
    W, H = landscape(letter)

    # Colors
    bg = HexColor("#070b14")
    card = HexColor("#0f1628")
    border = HexColor("#1c2a45")
    txt = HexColor("#e9eefc")
    muted = HexColor("#a7b4d6")
    accent = HexColor("#4aa3ff")
    good = HexColor("#29d07f")
    warn = HexColor("#f6c343")
    bad = HexColor("#ff5d6c")

    c.setFillColor(bg); c.rect(0, 0, W, H, stroke=0, fill=1)

    margin = 0.55 * inch
    gx = margin
    gy = margin
    gw = W - 2*margin
    gh = H - 2*margin

    # Header
    c.setFillColor(txt)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(gx, H - margin + 10, "ONE‑PAGER EJECUTIVO — EVALUACIÓN FINANCIERA (PYG)")
    c.setFont("Helvetica", 9)
    c.setFillColor(muted)
    c.drawRightString(W - margin, H - margin + 12, f"Fecha: {date.today().isoformat()}")
    c.drawString(gx, H - margin - 4, "USIL — MBA — Proyectos de Inversión / Valuation")

    # KPI row
    kpi_y = H - margin - 60
    kpi_h = 52
    kpi_gap = 10
    kpi_w = (gw - 5*kpi_gap) / 6

    def kpi_box(i, label, value, sub, tone="accent"):
        x = gx + i*(kpi_w + kpi_gap)
        y = kpi_y
        c.setFillColor(card); c.setStrokeColor(border); c.roundRect(x, y, kpi_w, kpi_h, 10, stroke=1, fill=1)
        c.setFillColor(muted); c.setFont("Helvetica", 7.5); c.drawString(x+10, y+kpi_h-16, label)
        c.setFillColor(txt); c.setFont("Helvetica-Bold", 10.5); c.drawString(x+10, y+kpi_h-32, value)
        c.setFillColor(muted); c.setFont("Helvetica", 7); c.drawString(x+10, y+10, sub)

    kpi_box(0, "VAN (base)", fmt_gs(d.npv_base), "Determinístico")
    kpi_box(1, "TIR (base)", fmt_pct(d.irr_base) if d.irr_base is not None else "N/A", "Determinístico")
    kpi_box(2, "Payback", f"{d.pb_simple:.2f} años" if d.pb_simple is not None else "N/A", "Simple")
    kpi_box(3, "Payback (desc.)", f"{d.pb_disc:.2f} años" if d.pb_disc is not None else "N/A", "Descontado")
    kpi_box(4, "P(VAN<0)", f"{d.prob_neg*100:.1f}%" if np.isfinite(d.prob_neg) else "N/A", "Monte Carlo")
    kpi_box(5, "P50 (VAN)", fmt_gs(d.p50) if np.isfinite(d.p50) else "N/A", "Monte Carlo")

    # Main grid: 2 rows x 2 cols
    top_y = kpi_y - 16
    card_h = 185
    col_gap = 14
    col_w = (gw - col_gap) / 2

    def card_box(x, y, w, h, title):
        c.setFillColor(card); c.setStrokeColor(border); c.roundRect(x, y, w, h, 14, stroke=1, fill=1)
        c.setFillColor(txt); c.setFont("Helvetica-Bold", 10); c.drawString(x+12, y+h-22, title)
        c.setStrokeColor(border); c.setLineWidth(1)
        c.line(x+12, y+h-28, x+w-12, y+h-28)

    # Card 1: Decision & assumptions
    x1 = gx
    y1 = top_y - card_h
    card_box(x1, y1, col_w, card_h, "Decisión & supuestos")
    c.setFillColor(muted); c.setFont("Helvetica", 8)
    c.drawString(x1+12, y1+card_h-46, f"Proyecto: {d.project} | Responsable: {d.responsible}")
    # Pill
    pill = d.verdict
    pill_color = good if pill=="APROBADO" else (warn if pill=="OBSERVADO" else bad)
    c.setFillColor(pill_color); c.roundRect(x1+12, y1+card_h-70, 120, 18, 9, stroke=0, fill=1)
    c.setFillColor(HexColor("#071014")); c.setFont("Helvetica-Bold", 8); c.drawString(x1+22, y1+card_h-66, f"DICTAMEN: {pill}")
    c.setFillColor(muted); c.setFont("Helvetica", 8)
    rationale = d.rationale
    # wrap
    lines = []
    for line in rationale.split("\n"):
        lines += textwrap.wrap(line, width=64)
    yy = y1+card_h-90
    for ln in lines[:4]:
        c.drawString(x1+12, yy, ln); yy -= 12

    c.setFillColor(muted); c.setFont("Helvetica", 7.8)
    items = [
        f"Horizonte: {d.n_years} años + perpetuidad",
        f"WACC: {fmt_pct(d.wacc)} (Ke {fmt_pct(d.ke)} | Kd {fmt_pct(d.kd)})",
        f"g explícito: {fmt_pct(d.g_exp)} | g∞: {fmt_pct(d.g_inf)}",
        f"CAPEX₀: {fmt_gs(d.capex0)} | FCF Año 1: {fmt_gs(d.fcf_y1)}",
    ]
    yy -= 2
    for it in items:
        c.drawString(x1+12, yy, "• " + it); yy -= 11

    # Card 2: Value structure
    x2 = gx + col_w + col_gap
    y2 = y1
    card_box(x2, y2, col_w, card_h, "Estructura del valor")
    c.setFillColor(muted); c.setFont("Helvetica", 7.8)
    c.drawString(x2+12, y2+card_h-46, "VAN = PV(FCF 1..N) + PV(TV) − CAPEX₀ (TV separado)")
    # Simple stacked bar
    bar_x, bar_y = x2+12, y2+70
    bar_w, bar_h = col_w-24, 18
    pv_f, pv_t = max(d.pv_fcf, 0), max(d.pv_tv, 0)
    total_pos = pv_f + pv_t
    if total_pos <= 0:
        total_pos = 1
    w_f = bar_w * (pv_f / total_pos)
    w_t = bar_w * (pv_t / total_pos)
    c.setFillColor(accent); c.rect(bar_x, bar_y, w_f, bar_h, stroke=0, fill=1)
    c.setFillColor(HexColor("#8fbfff")); c.rect(bar_x+w_f, bar_y, w_t, bar_h, stroke=0, fill=1)
    c.setStrokeColor(border); c.rect(bar_x, bar_y, bar_w, bar_h, stroke=1, fill=0)
    c.setFillColor(muted); c.setFont("Helvetica", 7)
    c.drawString(bar_x, bar_y+24, f"PV FCF: {fmt_gs(d.pv_fcf)}")
    c.drawString(bar_x, bar_y-12, f"PV TV:  {fmt_gs(d.pv_tv)}")
    c.drawRightString(x2+col_w-12, bar_y+6, f"CAPEX₀: {fmt_gs(d.capex0)}")
    c.setFillColor(txt); c.setFont("Helvetica-Bold", 10)
    c.drawString(x2+12, y2+42, f"VAN (base): {fmt_gs(d.npv_base)}")

    # Bottom row cards
    card_h2 = 210
    y3 = y1 - card_h2 - 14

    # Card 3: Cashflows
    card_box(x1, y3, col_w, card_h2, "Flujos proyectados (ciclo 1..N) + TV")
    # chart
    vals = list(d.fcf_explicit.tolist())
    labels = [str(int(i)) for i in range(1, d.n_years+1)]
    draw_bar_chart_pdf(c, x1+22, y3+60, col_w-44, 110, vals, labels, color="#4aa3ff")
    c.setFillColor(muted); c.setFont("Helvetica", 7.8)
    c.drawString(x1+12, y3+40, f"TV (en ciclo {d.n_years}): {fmt_gs(d.tv) if np.isfinite(d.tv) else 'N/A'}")

    # Card 4: Risk summary
    card_box(x2, y3, col_w, card_h2, "Riesgo (Monte Carlo) — lectura ejecutiva")

    c.setFillColor(muted); c.setFont("Helvetica", 8)
    bullet_lines = [
        f"Simulaciones: {d.sims:,}  |  válidas: {d.valid_rate*100:.1f}% (consistencia TV)",
        f"P(VAN<0): {d.prob_neg*100:.1f}%",
        f"Percentiles: P5 {fmt_gs(d.p5)} · P50 {fmt_gs(d.p50)} · P95 {fmt_gs(d.p95)}",
        f"Media {fmt_gs(d.mean_npv)} | Desv.Est. {fmt_gs(d.std_npv)} | CVaR5 {fmt_gs(d.cvar5)}",
    ]
    expl = "Lectura: P5≈downside plausible; P50≈escenario central; P95≈upside. CVaR5 resume el promedio del peor 5%."
    yy = y3+card_h2-46
    for ln in bullet_lines:
        c.drawString(x2+12, yy, "• " + ln); yy -= 12

    c.setFillColorRGB(0.78, 0.82, 0.90)
    for line in textwrap.wrap(expl, width=78):
        c.drawString(x2+12, yy, line); yy -= 11

    # Histograma (con P5/P50/P95)
    chart_x = x2 + 12
    chart_y = y3 + 14
    chart_w = col_w - 24
    chart_h = max(60, card_h2 - 120)
    draw_histogram_pdf(
        c, chart_x, chart_y, chart_w, chart_h,
        d.hist_edges, d.hist_counts,
        vlines=[(d.p5, (0.80,0.80,0.80), True), (d.p50, (1.00,1.00,1.00), False), (d.p95, (0.80,0.80,0.80), True)]
    )

    # footer
    c.setFillColor(muted); c.setFont("Helvetica", 7)
    c.drawString(gx, gy-6, "Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence.")
    c.drawRightString(W-margin, gy-6, "ValuationSuite USIL — One‑Pager (PYG)")

    c.showPage()
    c.save()
    return buf.getvalue()

# Export buttons
st.markdown("### Export")
colx1, colx2 = st.columns([1, 2], gap="small")


# --- One‑Pager extras (Monte Carlo) ---
_valid_npvs = npv_sims[np.isfinite(npv_sims)] if 'npv_sims' in locals() else np.array([])
if _valid_npvs.size > 0:
    hist_counts, hist_edges = np.histogram(_valid_npvs, bins=28)
else:
    hist_counts = np.array([])
    hist_edges = np.array([])

# Fallbacks defensivos
mean_npv = float(mean_npv) if 'mean_npv' in locals() and np.isfinite(mean_npv) else (float(np.mean(_valid_npvs)) if _valid_npvs.size>0 else np.nan)
std_npv  = float(std_npv)  if 'std_npv'  in locals() and np.isfinite(std_npv)  else (float(np.std(_valid_npvs, ddof=1)) if _valid_npvs.size>1 else 0.0)
skew_npv = float(skew_npv) if 'skew_npv' in locals() and np.isfinite(skew_npv) else np.nan
cvar5    = float(cvar5)    if 'cvar5'    in locals() and np.isfinite(cvar5)    else np.nan
valid_rate = float(valid_rate) if 'valid_rate' in locals() and np.isfinite(valid_rate) else (float(_valid_npvs.size / sims) if 'sims' in locals() and sims else np.nan)

onepager = OnePagerData(
    project=project,
    responsible=responsible,
    capex0=capex0,
    wacc=wacc, ke=ke, kd=kd,
    g_exp=g_exp, g_inf=g_inf,
    n_years=n_years,
    fcf_y1=fcf_y1,
    fcf_explicit=fcf_explicit,
    tv=tv,
    pv_fcf=pv_fcf,
    pv_tv=pv_tv_safe,
    npv_base=npv_base,
    irr_base=irr_base,
    pb_simple=pb_simple,
    pb_disc=pb_disc,
    sims=sims,
    valid_rate=valid_rate,
    prob_neg=prob_neg if np.isfinite(prob_neg) else 1.0,
    p5=p5 if np.isfinite(p5) else np.nan,
    p50=p50 if np.isfinite(p50) else np.nan,
    p95=p95 if np.isfinite(p95) else np.nan,
    verdict=verdict,
    rationale=rationale
)
with colx1:
    txt = f"""ONE‑PAGER EJECUTIVO — EVALUACIÓN FINANCIERA (PYG)
Fecha: {date.today().isoformat()}
Proyecto: {project}
Responsable: {responsible}

DICTAMEN: {verdict}
{rationale}

KPIs (determinístico)
- VAN: {fmt_gs(npv_base)}
- TIR: {fmt_pct(irr_base) if irr_base is not None else 'N/A'}
- Payback: {(f'{pb_simple:.2f} años') if pb_simple is not None else 'N/A'} | Payback descont.: {(f'{pb_disc:.2f} años') if pb_disc is not None else 'N/A'}

Flujos proyectados (FCF) — ciclo 1..N
""" + "\n".join([f"- Año {int(y)}: {fmt_gs(v)}" for y, v in zip(years, fcf_explicit)]) + f"""

Valor Terminal (TV, en ciclo {n_years}): {fmt_gs(tv) if np.isfinite(tv) else 'N/A'}

Riesgo (Monte Carlo)
- Simulaciones: {sims:,}
- P(VAN<0): {prob_neg*100:.1f}% 
- P5: {fmt_gs(p5)} | P50: {fmt_gs(p50)} | P95: {fmt_gs(p95)}

Nota: uso académico. No sustituye due diligence.
"""
    st.download_button("⬇️ Descargar One‑Pager (TXT)", data=txt.encode("utf-8"), file_name="one_pager_valuation_usil_PYG.txt")

with colx2:
    if REPORTLAB_OK:
        try:
            pdf_bytes = generate_onepager_pdf(onepager)
            st.download_button("⬇️ Descargar One‑Pager (PDF)", data=pdf_bytes, file_name="one_pager_valuation_usil_PYG.pdf", mime="application/pdf")
            st.caption("PDF 1 página (sin matplotlib).")
        except Exception as e:
            st.warning(f"No se pudo generar PDF: {e}")
    else:
        st.info("Para exportar PDF, agrega `reportlab` a requirements.txt."

def fmt_pct(x: float) -> str:
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "—"

def safe_float(x, default=0.0):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

def safe_irr(cashflows: np.ndarray):
    # npf.irr puede devolver NaN o valores extremos en casos numéricos raros
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
    # cashflows: array con CF0 negativo, CF1.. positivo
    cum = 0.0
    for t in range(len(cashflows)):
        cum += cashflows[t]
        if cum >= 0 and t > 0:
            prev_cum = cum - cashflows[t]
            # interpolación lineal dentro del año t
            if cashflows[t] == 0:
                return float(t)
            frac = (0 - prev_cum) / cashflows[t]
            return float(t - 1 + frac)
    return None

def payback_discounted(cashflows: np.ndarray, wacc: float):
    cum = 0.0
    for t in range(len(cashflows)):
        pv = cashflows[t] / ((1 + wacc) ** t)
        cum += pv
        if cum >= 0 and t > 0:
            prev_cum = cum - pv
            if pv == 0:
                return float(t)
            frac = (0 - prev_cum) / pv
            return float(t - 1 + frac)
    return None

def committee_verdict(npv_base, prob_neg, p5, p50):
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
        return "RECHAZADO", "El proyecto no cumple criterios mínimos del comité. Requiere rediseño de supuestos y mitigaciones.", checks
    return "OBSERVADO", "El proyecto muestra potencial, pero requiere evidencia/mitigaciones en supuestos críticos antes de aprobación.", checks

def wrap_lines(text: str, width=90):
    return textwrap.fill(text, width=width)

# ----------------------------
# Styling (premium dashboard)
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
.block-container {padding-top: 1.0rem; padding-bottom: 1.5rem; max-width: 1200px;}
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
  height: 94px;
}
.kpi .label{color:var(--muted); font-size:0.85rem; margin-bottom:6px;}
.kpi .value{color:var(--txt); font-size:1.25rem; font-weight:700; line-height:1.1;}
.kpi .sub{color:var(--muted); font-size:0.80rem; margin-top:6px;}
.pill{
  display:inline-block;
  padding:4px 10px;
  border-radius:999px;
  font-size:0.82rem;
  border:1px solid var(--border);
  color:var(--txt);
  background: rgba(255,255,255,0.03);
}
.pill.good{border-color: rgba(41,208,127,0.55); background: rgba(41,208,127,0.10);}
.pill.warn{border-color: rgba(246,195,67,0.55); background: rgba(246,195,67,0.10);}
.pill.bad{border-color: rgba(255,93,108,0.55); background: rgba(255,93,108,0.10);}
.hr{height:1px;background:var(--border); margin:10px 0 12px 0;}
.note{color:var(--muted); font-size:0.82rem;}

.headerbar{display:flex;justify-content:space-between;align-items:flex-end;gap:14px;flex-wrap:wrap}
.hb-title{font-size:18px;font-weight:800;letter-spacing:.2px;line-height:1.2;margin-bottom:4px}
.hb-sub{font-size:12px;opacity:.85;line-height:1.35}
.hb-right{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;justify-content:flex-end}
.pill{padding:6px 10px;border:1px solid rgba(255,255,255,.12);border-radius:999px;background:rgba(255,255,255,.04);font-size:12px}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
today_str = date.today().isoformat()

st.markdown(
    f"""
    <div class="card headerbar">
      <div class="hb-left">
        <div class="hb-title">ONE-PAGER EJECUTIVO — EVALUACIÓN FINANCIERA</div>
        <div class="hb-sub">
          Universidad San Ignacio de Loyola (USIL) — Maestría en Administración de Negocios (MBA) — Proyectos de Inversión / Valuation
        </div>
      </div>
      <div class="hb-right">
        <div class="pill">Moneda: <b>PYG</b></div>
        <div class="pill">Fecha: <b>{today_str}</b></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
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

    # WACC alrededor del calculado (±2%)
    w_spread = st.number_input("WACC rango ± (%)", value=2.0, step=0.25) / 100

    capex_min = st.number_input("CAPEX mín (favorable)", value=capex0 * 0.90, step=10_000_000.0)
    capex_mode = st.number_input("CAPEX base", value=capex0, step=10_000_000.0)
    capex_max = st.number_input("CAPEX máx (adverso)", value=capex0 * 1.15, step=10_000_000.0)

    fcf_mult_min = st.number_input("Shock FCF1 mín (adverso)", value=0.85, step=0.01)
    fcf_mult_mode = st.number_input("Shock FCF1 base", value=1.00, step=0.01)
    fcf_mult_max = st.number_input("Shock FCF1 máx (favorable)", value=1.12, step=0.01)

# ----------------------------
# Cálculos determinísticos
# ----------------------------
# CAPM + WACC
beta_l = beta_u * (1 + (1 - tax_rate) * (debt / max(equity, 1e-9)))
ke = rf + beta_l * erp + crp

V = debt + equity
wd = (debt / V) if V > 0 else 0.0
we = (equity / V) if V > 0 else 1.0
wacc = we * ke + wd * kd * (1 - tax_rate)

# Contable Año 1 -> FCF Año 1
ebit = sales - var_costs - fixed_costs - depreciation
nopat = ebit * (1 - tax_rate)
delta_nwc = (delta_ar + delta_inv - delta_ap)  # aumento de NWC => consumo de caja
fcf_y1 = nopat + depreciation - capex_y1 - delta_nwc

# Flujos explícitos ciclo 1..N (desde FCF año 1)
years = np.arange(1, n_years + 1)
fcf_explicit = np.array([fcf_y1 * ((1 + g_exp) ** (t - 1)) for t in years], dtype=float)

# Valor terminal (Gordon)
tv_ok = wacc > (g_inf + MIN_SPREAD)
tv = (fcf_explicit[-1] * (1 + g_inf) / (wacc - g_inf)) if tv_ok else np.nan

# PV de flujos y TV
disc = (1 + wacc) ** years
pv_fcf = float(np.sum(fcf_explicit / disc))
pv_tv = float(tv / ((1 + wacc) ** n_years)) if np.isfinite(tv) else np.nan
npv_base = pv_fcf + (pv_tv if np.isfinite(pv_tv) else 0.0) - capex0

# TIR base (incluye TV como parte del último flujo)
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
    # g triangular
    g_min: float, g_mode: float, g_max: float,
    # wacc triangular (derivado)
    w_min: float, w_mode: float, w_max: float,
    # capex triangular
    capex_min: float, capex_mode: float, capex_max: float,
    # shock FCF1 triangular
    m_min: float, m_mode: float, m_max: float,
):
    rng = np.random.default_rng()

    g_s = rng.triangular(g_min, g_mode, g_max, sims)
    w_s = rng.triangular(w_min, w_mode, w_max, sims)
    capex_s = rng.triangular(capex_min, capex_mode, capex_max, sims)
    m_s = rng.triangular(m_min, m_mode, m_max, sims)

    yrs = np.arange(1, n_years + 1)
    fcf1 = fcf_y1 * m_s
    fcf_paths = fcf1[:, None] * (1.0 + g_s)[:, None] ** (yrs[None, :] - 1)

    valid = w_s > (g_inf + min_spread)
    npv_s = np.full(sims, np.nan)

    idx = np.where(valid)[0]
    if idx.size == 0:
        return npv_s, g_s, w_s, capex_s, m_s, idx

    fcf_v = fcf_paths[idx, :].copy()
    w_v = w_s[idx]
    cap_v = capex_s[idx]

    tv_v = (fcf_v[:, -1] * (1 + g_inf)) / (w_v - g_inf)
    fcf_v[:, -1] += tv_v

    disc_v = (1 + w_v)[:, None] ** (yrs[None, :])
    pv = np.sum(fcf_v / disc_v, axis=1)
    npv_s[idx] = pv - cap_v
    return npv_s, g_s, w_s, capex_s, m_s, idx

w_min = max(0.0001, wacc - w_spread)
w_mode = max(0.0001, wacc)
w_max = max(w_min + 0.0001, wacc + w_spread)

npv_sims, g_s, w_s, cap_s, m_s, valid_idx = run_monte_carlo(
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
if valid_npvs.size == 0:
    prob_neg = np.nan
    p5 = p50 = p95 = np.nan
    mean_npv = std_npv = skew_npv = cvar5 = valid_rate = np.nan
else:
    prob_neg = float(np.mean(valid_npvs < 0))
    p5, p50, p95 = np.percentile(valid_npvs, [5, 50, 95])
    mean_npv = float(np.mean(valid_npvs))
    std_npv = float(np.std(valid_npvs, ddof=1)) if valid_npvs.size > 1 else 0.0
    skew_npv = float(np.mean(((valid_npvs-mean_npv)/(std_npv if std_npv>0 else 1.0))**3)) if valid_npvs.size > 2 else 0.0
    cvar5 = float(np.mean(valid_npvs[valid_npvs <= p5])) if valid_npvs.size > 0 else np.nan
    valid_rate = float(np.mean(np.isfinite(npv_sims)))

verdict, rationale, checks = committee_verdict(npv_base=npv_base, prob_neg=prob_neg if np.isfinite(prob_neg) else 1.0, p5=p5 if np.isfinite(p5) else -1e18, p50=p50 if np.isfinite(p50) else -1e18)

def pill_class(v):
    return "good" if v=="APROBADO" else ("warn" if v=="OBSERVADO" else "bad")

# ----------------------------
# KPI row
# ----------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")

with k1:
    st.markdown(f"<div class='kpi'><div class='label'>VAN (base)</div><div class='value'>{fmt_gs(npv_base)}</div><div class='sub'>Determinístico</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi'><div class='label'>TIR (base)</div><div class='value'>{fmt_pct(irr_base) if irr_base is not None else 'N/A'}</div><div class='sub'>Determinístico</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi'><div class='label'>Payback</div><div class='value'>{(f'{pb_simple:.2f} años') if pb_simple is not None else 'N/A'}</div><div class='sub'>Simple</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi'><div class='label'>Payback (desc.)</div><div class='value'>{(f'{pb_disc:.2f} años') if pb_disc is not None else 'N/A'}</div><div class='sub'>Descontado</div></div>", unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='kpi'><div class='label'>P(VAN&lt;0)</div><div class='value'>{(f'{prob_neg*100:.1f}%') if np.isfinite(prob_neg) else 'N/A'}</div><div class='sub'>Monte Carlo</div></div>", unsafe_allow_html=True)
with k6:
    st.markdown(f"<div class='kpi'><div class='label'>P50 (VAN)</div><div class='value'>{fmt_gs(p50) if np.isfinite(p50) else 'N/A'}</div><div class='sub'>Monte Carlo</div></div>", unsafe_allow_html=True)

# ----------------------------
# Middle row (2 cards) — acá estaba “corto”: lo hacemos más lleno y balanceado
# ----------------------------
c_left, c_right = st.columns([1.08, 0.92], gap="large")

with c_left:
    # Card: decisión + supuestos + checklist
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Decisión & supuestos")
    st.markdown(
        f"<span class='pill {pill_class(verdict)}'>DICTAMEN: {verdict}</span>",
        unsafe_allow_html=True
    )
    st.markdown(f"<div class='note'>{rationale}</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    colA, colB = st.columns([1,1], gap="small")
    with colA:
        st.markdown("**Supuestos clave**")
        st.markdown(
            f"- Horizonte: **{n_years} años** + perpetuidad\n"
            f"- g explícito: **{fmt_pct(g_exp)}**  |  g∞: **{fmt_pct(g_inf)}**\n"
            f"- WACC: **{fmt_pct(wacc)}**  (Ke {fmt_pct(ke)} | Kd {fmt_pct(kd)})\n"
            f"- CAPEX 0: **{fmt_gs(capex0)}**\n"
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
    st.markdown("**Lectura ejecutiva (1 párrafo)**")
    executive_read = (
        f"Con {n_years} años explícitos y crecimiento a perpetuidad g∞={fmt_pct(g_inf)}, el VAN base es {fmt_gs(npv_base)} "
        f"y la TIR base {fmt_pct(irr_base) if irr_base is not None else 'N/A'}. "
        f"El downside bajo Monte Carlo se resume en P(VAN<0)={prob_neg*100:.1f}% y P5={fmt_gs(p5)}. "
        f"Conclusión: {verdict}."
    )
    st.markdown(f"<div class='note'>{executive_read}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c_right:
    # Card: estructura del valor (PV FCF, PV TV, CAPEX) y mini “waterfall”
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Estructura del valor")
    st.markdown("<div class='note'>VAN = PV(FCF ciclo 1..N) + PV(TV) − CAPEX₀. TV se reporta separado para transparencia.</div>", unsafe_allow_html=True)

    pv_tv_safe = pv_tv if np.isfinite(pv_tv) else 0.0
    data = pd.DataFrame({
        "Componente": ["PV FCF", "PV TV", "- CAPEX₀"],
        "Valor": [pv_fcf, pv_tv_safe, -capex0]
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data["Componente"],
        y=data["Valor"],
        text=[fmt_gs(v) for v in data["Valor"]],
        textposition="outside",
        hovertemplate="%{x}<br>%{text}<extra></extra>",
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
        st.warning("Inconsistencia TV: WACC debe ser mayor que g∞ + 0.5%. Ajusta supuestos.", icon="⚠️")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Bottom row (2 charts)
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
    st.markdown("### Riesgo (Monte Carlo) — lectura ejecutiva")
    if valid_npvs.size == 0:
        st.error("No hay simulaciones válidas (WACC ≤ g∞ + spread). Ajusta supuestos.")
    else:
        st.markdown(
            f"<div class='note'>P5: <b>{fmt_gs(p5)}</b> · P50: <b>{fmt_gs(p50)}</b> · P95: <b>{fmt_gs(p95)}</b> · "
            f"P(VAN&lt;0): <b>{prob_neg*100:.1f}%</b> · Simulaciones: <b>{sims:,}</b></div>",
            unsafe_allow_html=True
        )
        df_hist = pd.DataFrame({"VAN": valid_npvs})
        fig_hist = px.histogram(df_hist, x="VAN", nbins=40)
        fig_hist.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="count", showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
            xaxis=dict(title="VAN"),
            showlegend=False
        )
        # Líneas P5, P50, P95
        for val, name in [(p5, "P5"), (p50, "P50"), (p95, "P95")]:
            fig_hist.add_vline(x=val, line_width=2, line_dash="dash", annotation_text=name, annotation_position="top")
        st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("<div class='note'>Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence. · ValuationSuite USIL (PYG)</div>", unsafe_allow_html=True)

# ----------------------------
# Export: One‑Pager PDF (1 página, sin matplotlib)
# ----------------------------
@dataclass
class OnePagerData:
    project: str
    responsible: str

    # Determinístico
    capex0: float
    wacc: float
    ke: float
    kd: float
    g_exp: float
    g_inf: float
    n_years: int
    fcf_y1: float
    fcf_explicit: np.ndarray
    tv: float
    pv_fcf: float
    pv_tv: float
    npv_base: float
    irr_base: float | None
    pb_simple: float | None
    pb_disc: float | None

    # Monte Carlo
    sims: int
    valid_rate: float
    prob_neg: float
    p5: float
    p50: float
    p95: float
    mean_npv: float
    std_npv: float
    skew_npv: float
    cvar5: float
    hist_edges: np.ndarray
    hist_counts: np.ndarray

    # Comité
    verdict: str
    rationale: str

def draw_bar_chart_pdf(c: canvas.Canvas, x, y, w, h, values, labels, color="#4aa3ff"):
    if len(values) == 0:
        return
    vmax = max(values) if max(values) != 0 else 1
    bar_w = w / (len(values) * 1.2)
    gap = bar_w * 0.2
    for i, v in enumerate(values):
        bh = (v / vmax) * h
        bx = x + i * (bar_w + gap)
        by = y
        c.setFillColor(HexColor(color))
        c.rect(bx, by, bar_w, bh, stroke=0, fill=1)
        c.setFillColor(HexColor("#a7b4d6"))
        c.setFont("Helvetica", 7)
        c.drawCentredString(bx + bar_w/2, y - 10, str(labels[i]))

def draw_histogram_pdf(c, x, y, w, h, edges, counts, vlines=None):
    """Histograma minimalista para el One‑Pager PDF (sin matplotlib)."""
    if edges is None or counts is None:
        return
    edges = np.asarray(edges, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0:
        return

    bins = int(counts.size)
    max_c = float(np.max(counts)) if np.max(counts) > 0 else 1.0

    # marco
    c.setStrokeColorRGB(0.35, 0.40, 0.50)
    c.rect(x, y, w, h, stroke=1, fill=0)

    bw = w / bins
    c.setFillColorRGB(0.43, 0.74, 1.00)
    for i in range(bins):
        bh = (float(counts[i]) / max_c) * (h - 6)
        c.rect(x + i*bw, y, bw*0.92, bh, stroke=0, fill=1)

    if vlines:
        xmin = float(edges[0]); xmax = float(edges[-1])
        span = (xmax - xmin) if (xmax - xmin) != 0 else 1.0
        for val, rgb, dashed in vlines:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            t = (float(val) - xmin) / span
            t = max(0.0, min(1.0, t))
            xx = x + t*w
            if rgb is None:
                rgb = (1.0, 1.0, 1.0)
            c.setStrokeColorRGB(*rgb)
            if dashed:
                c.setDash(4, 3)
            else:
                c.setDash()
            c.line(xx, y, xx, y + h)
        c.setDash()


def generate_onepager_pdf(d: OnePagerData) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab no está disponible. Agrega `reportlab` a requirements.txt.")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(letter))
    W, H = landscape(letter)

    # Colors
    bg = HexColor("#070b14")
    card = HexColor("#0f1628")
    border = HexColor("#1c2a45")
    txt = HexColor("#e9eefc")
    muted = HexColor("#a7b4d6")
    accent = HexColor("#4aa3ff")
    good = HexColor("#29d07f")
    warn = HexColor("#f6c343")
    bad = HexColor("#ff5d6c")

    c.setFillColor(bg); c.rect(0, 0, W, H, stroke=0, fill=1)

    margin = 0.55 * inch
    gx = margin
    gy = margin
    gw = W - 2*margin
    gh = H - 2*margin

    # Header
    c.setFillColor(txt)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(gx, H - margin + 10, "ONE‑PAGER EJECUTIVO — EVALUACIÓN FINANCIERA (PYG)")
    c.setFont("Helvetica", 9)
    c.setFillColor(muted)
    c.drawRightString(W - margin, H - margin + 12, f"Fecha: {date.today().isoformat()}")
    c.drawString(gx, H - margin - 4, "USIL — MBA — Proyectos de Inversión / Valuation")

    # KPI row
    kpi_y = H - margin - 60
    kpi_h = 52
    kpi_gap = 10
    kpi_w = (gw - 5*kpi_gap) / 6

    def kpi_box(i, label, value, sub, tone="accent"):
        x = gx + i*(kpi_w + kpi_gap)
        y = kpi_y
        c.setFillColor(card); c.setStrokeColor(border); c.roundRect(x, y, kpi_w, kpi_h, 10, stroke=1, fill=1)
        c.setFillColor(muted); c.setFont("Helvetica", 7.5); c.drawString(x+10, y+kpi_h-16, label)
        c.setFillColor(txt); c.setFont("Helvetica-Bold", 10.5); c.drawString(x+10, y+kpi_h-32, value)
        c.setFillColor(muted); c.setFont("Helvetica", 7); c.drawString(x+10, y+10, sub)

    kpi_box(0, "VAN (base)", fmt_gs(d.npv_base), "Determinístico")
    kpi_box(1, "TIR (base)", fmt_pct(d.irr_base) if d.irr_base is not None else "N/A", "Determinístico")
    kpi_box(2, "Payback", f"{d.pb_simple:.2f} años" if d.pb_simple is not None else "N/A", "Simple")
    kpi_box(3, "Payback (desc.)", f"{d.pb_disc:.2f} años" if d.pb_disc is not None else "N/A", "Descontado")
    kpi_box(4, "P(VAN<0)", f"{d.prob_neg*100:.1f}%" if np.isfinite(d.prob_neg) else "N/A", "Monte Carlo")
    kpi_box(5, "P50 (VAN)", fmt_gs(d.p50) if np.isfinite(d.p50) else "N/A", "Monte Carlo")

    # Main grid: 2 rows x 2 cols
    top_y = kpi_y - 16
    card_h = 185
    col_gap = 14
    col_w = (gw - col_gap) / 2

    def card_box(x, y, w, h, title):
        c.setFillColor(card); c.setStrokeColor(border); c.roundRect(x, y, w, h, 14, stroke=1, fill=1)
        c.setFillColor(txt); c.setFont("Helvetica-Bold", 10); c.drawString(x+12, y+h-22, title)
        c.setStrokeColor(border); c.setLineWidth(1)
        c.line(x+12, y+h-28, x+w-12, y+h-28)

    # Card 1: Decision & assumptions
    x1 = gx
    y1 = top_y - card_h
    card_box(x1, y1, col_w, card_h, "Decisión & supuestos")
    c.setFillColor(muted); c.setFont("Helvetica", 8)
    c.drawString(x1+12, y1+card_h-46, f"Proyecto: {d.project} | Responsable: {d.responsible}")
    # Pill
    pill = d.verdict
    pill_color = good if pill=="APROBADO" else (warn if pill=="OBSERVADO" else bad)
    c.setFillColor(pill_color); c.roundRect(x1+12, y1+card_h-70, 120, 18, 9, stroke=0, fill=1)
    c.setFillColor(HexColor("#071014")); c.setFont("Helvetica-Bold", 8); c.drawString(x1+22, y1+card_h-66, f"DICTAMEN: {pill}")
    c.setFillColor(muted); c.setFont("Helvetica", 8)
    rationale = d.rationale
    # wrap
    lines = []
    for line in rationale.split("\n"):
        lines += textwrap.wrap(line, width=64)
    yy = y1+card_h-90
    for ln in lines[:4]:
        c.drawString(x1+12, yy, ln); yy -= 12

    c.setFillColor(muted); c.setFont("Helvetica", 7.8)
    items = [
        f"Horizonte: {d.n_years} años + perpetuidad",
        f"WACC: {fmt_pct(d.wacc)} (Ke {fmt_pct(d.ke)} | Kd {fmt_pct(d.kd)})",
        f"g explícito: {fmt_pct(d.g_exp)} | g∞: {fmt_pct(d.g_inf)}",
        f"CAPEX₀: {fmt_gs(d.capex0)} | FCF Año 1: {fmt_gs(d.fcf_y1)}",
    ]
    yy -= 2
    for it in items:
        c.drawString(x1+12, yy, "• " + it); yy -= 11

    # Card 2: Value structure
    x2 = gx + col_w + col_gap
    y2 = y1
    card_box(x2, y2, col_w, card_h, "Estructura del valor")
    c.setFillColor(muted); c.setFont("Helvetica", 7.8)
    c.drawString(x2+12, y2+card_h-46, "VAN = PV(FCF 1..N) + PV(TV) − CAPEX₀ (TV separado)")
    # Simple stacked bar
    bar_x, bar_y = x2+12, y2+70
    bar_w, bar_h = col_w-24, 18
    pv_f, pv_t = max(d.pv_fcf, 0), max(d.pv_tv, 0)
    total_pos = pv_f + pv_t
    if total_pos <= 0:
        total_pos = 1
    w_f = bar_w * (pv_f / total_pos)
    w_t = bar_w * (pv_t / total_pos)
    c.setFillColor(accent); c.rect(bar_x, bar_y, w_f, bar_h, stroke=0, fill=1)
    c.setFillColor(HexColor("#8fbfff")); c.rect(bar_x+w_f, bar_y, w_t, bar_h, stroke=0, fill=1)
    c.setStrokeColor(border); c.rect(bar_x, bar_y, bar_w, bar_h, stroke=1, fill=0)
    c.setFillColor(muted); c.setFont("Helvetica", 7)
    c.drawString(bar_x, bar_y+24, f"PV FCF: {fmt_gs(d.pv_fcf)}")
    c.drawString(bar_x, bar_y-12, f"PV TV:  {fmt_gs(d.pv_tv)}")
    c.drawRightString(x2+col_w-12, bar_y+6, f"CAPEX₀: {fmt_gs(d.capex0)}")
    c.setFillColor(txt); c.setFont("Helvetica-Bold", 10)
    c.drawString(x2+12, y2+42, f"VAN (base): {fmt_gs(d.npv_base)}")

    # Bottom row cards
    card_h2 = 210
    y3 = y1 - card_h2 - 14

    # Card 3: Cashflows
    card_box(x1, y3, col_w, card_h2, "Flujos proyectados (ciclo 1..N) + TV")
    # chart
    vals = list(d.fcf_explicit.tolist())
    labels = [str(int(i)) for i in range(1, d.n_years+1)]
    draw_bar_chart_pdf(c, x1+22, y3+60, col_w-44, 110, vals, labels, color="#4aa3ff")
    c.setFillColor(muted); c.setFont("Helvetica", 7.8)
    c.drawString(x1+12, y3+40, f"TV (en ciclo {d.n_years}): {fmt_gs(d.tv) if np.isfinite(d.tv) else 'N/A'}")

    # Card 4: Risk summary
    card_box(x2, y3, col_w, card_h2, "Riesgo (Monte Carlo) — lectura ejecutiva")

    c.setFillColor(muted); c.setFont("Helvetica", 8)
    bullet_lines = [
        f"Simulaciones: {d.sims:,}  |  válidas: {d.valid_rate*100:.1f}% (consistencia TV)",
        f"P(VAN<0): {d.prob_neg*100:.1f}%",
        f"Percentiles: P5 {fmt_gs(d.p5)} · P50 {fmt_gs(d.p50)} · P95 {fmt_gs(d.p95)}",
        f"Media {fmt_gs(d.mean_npv)} | Desv.Est. {fmt_gs(d.std_npv)} | CVaR5 {fmt_gs(d.cvar5)}",
    ]
    expl = "Lectura: P5≈downside plausible; P50≈escenario central; P95≈upside. CVaR5 resume el promedio del peor 5%."
    yy = y3+card_h2-46
    for ln in bullet_lines:
        c.drawString(x2+12, yy, "• " + ln); yy -= 12

    c.setFillColorRGB(0.78, 0.82, 0.90)
    for line in textwrap.wrap(expl, width=78):
        c.drawString(x2+12, yy, line); yy -= 11

    # Histograma (con P5/P50/P95)
    chart_x = x2 + 12
    chart_y = y3 + 14
    chart_w = col_w - 24
    chart_h = max(60, card_h2 - 120)
    draw_histogram_pdf(
        c, chart_x, chart_y, chart_w, chart_h,
        d.hist_edges, d.hist_counts,
        vlines=[(d.p5, (0.80,0.80,0.80), True), (d.p50, (1.00,1.00,1.00), False), (d.p95, (0.80,0.80,0.80), True)]
    )

    # footer
    c.setFillColor(muted); c.setFont("Helvetica", 7)
    c.drawString(gx, gy-6, "Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence.")
    c.drawRightString(W-margin, gy-6, "ValuationSuite USIL — One‑Pager (PYG)")

    c.showPage()
    c.save()
    return buf.getvalue()

# Export buttons
st.markdown("### Export")
colx1, colx2 = st.columns([1, 2], gap="small")


# --- One‑Pager extras (Monte Carlo) ---
_valid_npvs = npv_sims[np.isfinite(npv_sims)] if 'npv_sims' in locals() else np.array([])
if _valid_npvs.size > 0:
    hist_counts, hist_edges = np.histogram(_valid_npvs, bins=28)
else:
    hist_counts = np.array([])
    hist_edges = np.array([])

# Fallbacks defensivos
mean_npv = float(mean_npv) if 'mean_npv' in locals() and np.isfinite(mean_npv) else (float(np.mean(_valid_npvs)) if _valid_npvs.size>0 else np.nan)
std_npv  = float(std_npv)  if 'std_npv'  in locals() and np.isfinite(std_npv)  else (float(np.std(_valid_npvs, ddof=1)) if _valid_npvs.size>1 else 0.0)
skew_npv = float(skew_npv) if 'skew_npv' in locals() and np.isfinite(skew_npv) else np.nan
cvar5    = float(cvar5)    if 'cvar5'    in locals() and np.isfinite(cvar5)    else np.nan
valid_rate = float(valid_rate) if 'valid_rate' in locals() and np.isfinite(valid_rate) else (float(_valid_npvs.size / sims) if 'sims' in locals() and sims else np.nan)

onepager = OnePagerData(
    project=project,
    responsible=responsible,
    capex0=capex0,
    wacc=wacc, ke=ke, kd=kd,
    g_exp=g_exp, g_inf=g_inf,
    n_years=n_years,
    fcf_y1=fcf_y1,
    fcf_explicit=fcf_explicit,
    tv=tv,
    pv_fcf=pv_fcf,
    pv_tv=pv_tv_safe,
    npv_base=npv_base,
    irr_base=irr_base,
    pb_simple=pb_simple,
    pb_disc=pb_disc,
    sims=sims,
    valid_rate=valid_rate,
    prob_neg=prob_neg if np.isfinite(prob_neg) else 1.0,
    p5=p5 if np.isfinite(p5) else np.nan,
    p50=p50 if np.isfinite(p50) else np.nan,
    p95=p95 if np.isfinite(p95) else np.nan,
    verdict=verdict,
    rationale=rationale
)
with colx1:
    txt = f"""ONE‑PAGER EJECUTIVO — EVALUACIÓN FINANCIERA (PYG)
Fecha: {date.today().isoformat()}
Proyecto: {project}
Responsable: {responsible}

DICTAMEN: {verdict}
{rationale}

KPIs (determinístico)
- VAN: {fmt_gs(npv_base)}
- TIR: {fmt_pct(irr_base) if irr_base is not None else 'N/A'}
- Payback: {(f'{pb_simple:.2f} años') if pb_simple is not None else 'N/A'} | Payback descont.: {(f'{pb_disc:.2f} años') if pb_disc is not None else 'N/A'}

Flujos proyectados (FCF) — ciclo 1..N
""" + "\n".join([f"- Año {int(y)}: {fmt_gs(v)}" for y, v in zip(years, fcf_explicit)]) + f"""

Valor Terminal (TV, en ciclo {n_years}): {fmt_gs(tv) if np.isfinite(tv) else 'N/A'}

Riesgo (Monte Carlo)
- Simulaciones: {sims:,}
- P(VAN<0): {prob_neg*100:.1f}% 
- P5: {fmt_gs(p5)} | P50: {fmt_gs(p50)} | P95: {fmt_gs(p95)}

Nota: uso académico. No sustituye due diligence.
"""
    st.download_button("⬇️ Descargar One‑Pager (TXT)", data=txt.encode("utf-8"), file_name="one_pager_valuation_usil_PYG.txt")

with colx2:
    if REPORTLAB_OK:
        try:
            pdf_bytes = generate_onepager_pdf(onepager)
            st.download_button("⬇️ Descargar One‑Pager (PDF)", data=pdf_bytes, file_name="one_pager_valuation_usil_PYG.pdf", mime="application/pdf")
            st.caption("PDF 1 página (sin matplotlib).")
        except Exception as e:
            st.warning(f"No se pudo generar PDF: {e}")
    else:
        st.info("Para exportar PDF, agrega `reportlab` a requirements.txt.")