# app.py — ValuationSuite USIL (PYG)
# Streamlit one-pager dashboard + PDF export (premium)
# Author: generated with ChatGPT for Jorge Rojas (USIL MBA)

import io
import math
import textwrap
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy_financial as npf
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# Optional deps for PDF
# -----------------------------
REPORTLAB_OK = True
MATPLOTLIB_OK = True
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.lib import colors
except Exception:
    REPORTLAB_OK = False

try:
    import matplotlib.pyplot as plt
except Exception:
    MATPLOTLIB_OK = False

# -----------------------------
# Config
# -----------------------------
MIN_SPREAD = 0.005  # WACC must be > g∞ + 0.5%
DEFAULT_SIMS = 15000

st.set_page_config(page_title="ValuationSuite USIL (PYG)", layout="wide")

# -----------------------------
# Styling (premium dashboard)
# -----------------------------
st.markdown(
    """
<style>
:root{
  --bg:#070B14;
  --card:#0D1626;
  --card2:#0B1322;
  --stroke:rgba(255,255,255,.08);
  --text:rgba(255,255,255,.92);
  --muted:rgba(255,255,255,.65);
  --accent:#38bdf8;
  --accent2:#22c55e;
}
.stApp{ background: radial-gradient(1200px 700px at 15% 10%, rgba(56,189,248,.18), transparent 55%),
                 radial-gradient(900px 600px at 85% 30%, rgba(34,197,94,.12), transparent 60%),
                 linear-gradient(180deg, #060913, #050712 70%, #040510) !important; }
.block-container{ padding-top: 1.2rem; padding-bottom: 2.0rem; max-width: 1250px; }
h1,h2,h3{ color: var(--text); }
.small-muted{ color: var(--muted); font-size: .9rem; }

/* Card */
.card{
  background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 14px 14px 10px 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
}
.card h4{ margin:0 0 6px 0; font-size: .95rem; color: var(--muted); font-weight: 600; }
.card .big{ font-size: 1.35rem; font-weight: 800; color: var(--text); line-height: 1.1; }
.card .sub{ font-size:.86rem; color: var(--muted); margin-top: 2px; }

.pill{
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,.03);
  font-size: .85rem;
  color: var(--muted);
}
.pill.ok{ background: rgba(34,197,94,.12); color: rgba(203,255,226,.95); border-color: rgba(34,197,94,.25); }
.pill.warn{ background: rgba(245,158,11,.12); color: rgba(255,236,201,.95); border-color: rgba(245,158,11,.25); }
.pill.bad{ background: rgba(239,68,68,.10); color: rgba(255,210,210,.95); border-color: rgba(239,68,68,.25); }

.hr{
  height:1px; background: var(--stroke); margin: 10px 0 12px 0;
}
/* Sidebar */
section[data-testid="stSidebar"]{
  background: rgba(13,22,38,.65);
  border-right: 1px solid var(--stroke);
}
/* Reduce Plotly background */
.js-plotly-plot .plotly .main-svg{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def fmt_pyg(x: float) -> str:
    try:
        x = float(x)
    except Exception:
        return "Gs. —"
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}Gs. {x:,.0f}".replace(",", ".")

def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x*100:.2f}%"

def safe_float(x, default=0.0):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

def safe_irr(cashflows: list[float]):
    try:
        irr = float(npf.irr(cashflows))
        if np.isnan(irr) or np.isinf(irr):
            return None
        # guardrail: unrealistic outputs often indicate multiple roots/ill-conditioned
        if irr < -0.99 or irr > 2.0:
            return None
        return irr
    except Exception:
        return None

def payback_simple(cashflows: list[float]):
    # cashflows: t0..tN, t0 negative
    cum = 0.0
    for t, cf in enumerate(cashflows):
        cum += cf
        if t == 0:
            continue
        if cum >= 0:
            prev = cum - cf
            if cf == 0:
                return float(t)
            frac = (0 - prev) / cf
            return (t - 1) + frac
    return None

def payback_discounted(cashflows: list[float], r: float):
    cum = 0.0
    for t, cf in enumerate(cashflows):
        if t == 0:
            cum += cf
            continue
        pv = cf / ((1 + r) ** t)
        cum += pv
        if cum >= 0:
            prev = cum - pv
            if pv == 0:
                return float(t)
            frac = (0 - prev) / pv
            return (t - 1) + frac
    return None

def verdict_from_checks(van_ok, pneg_ok, p50_ok, p5_ok):
    n_ok = sum([van_ok, pneg_ok, p50_ok, p5_ok])
    if n_ok == 4:
        return "APROBADO", "ok"
    if n_ok <= 1:
        return "RECHAZADO", "bad"
    return "OBSERVADO", "warn"

def triangular(rng, a, m, b, n):
    a, m, b = float(a), float(m), float(b)
    if not (a <= m <= b):
        # fallback to a symmetric-ish triangle
        m = min(max(m, a), b)
    return rng.triangular(a, m, b, n)

# -----------------------------
# Monte Carlo engine
# -----------------------------
@st.cache_data(show_spinner=False)
def run_monte_carlo(
    sims: int,
    fcf1: float,
    n_years: int,
    g_inf: float,
    wacc_base: float,
    wacc_pm: float,
    g_min: float, g_mode: float, g_max: float,
    capex_min: float, capex_mode: float, capex_max: float,
    fcf_mult_min: float, fcf_mult_mode: float, fcf_mult_max: float,
):
    rng = np.random.default_rng()  # internal seed (not exposed)

    g_s = triangular(rng, g_min, g_mode, g_max, sims)
    w_s = triangular(rng, wacc_base * (1 - wacc_pm), wacc_base, wacc_base * (1 + wacc_pm), sims)
    capex0_s = triangular(rng, capex_min, capex_mode, capex_max, sims)
    mult_s = triangular(rng, fcf_mult_min, fcf_mult_mode, fcf_mult_max, sims)

    yrs = np.arange(1, n_years + 1, dtype=float)
    fcf1_s = fcf1 * mult_s
    fcf_paths = fcf1_s[:, None] * (1 + g_s)[:, None] ** (yrs[None, :] - 1)

    valid = w_s > (g_inf + MIN_SPREAD)
    npv = np.full(sims, np.nan, dtype=float)

    idx = np.where(valid)[0]
    if idx.size == 0:
        return npv, valid

    fcf_v = fcf_paths[idx, :]
    w_v = w_s[idx]
    capex0_v = capex0_s[idx]

    tv = (fcf_v[:, -1] * (1 + g_inf)) / (w_v - g_inf)
    fcf_v[:, -1] = fcf_v[:, -1] + tv

    disc = (1 + w_v)[:, None] ** (yrs[None, :])
    pv = np.sum(fcf_v / disc, axis=1)
    npv[idx] = pv - capex0_v

    return npv, valid

# -----------------------------
# PDF helpers
# -----------------------------
def _mpl_value_bridge(pv_fcf, pv_tv, capex0):
    fig, ax = plt.subplots(figsize=(4.0, 1.8), dpi=200)
    ax.barh(["VAN"], [pv_fcf], label="PV FCF")
    ax.barh(["VAN"], [pv_tv], left=[pv_fcf], label="PV TV")
    ax.barh(["VAN"], [-capex0], left=[pv_fcf + pv_tv], label="CAPEX₀")
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Gs.")
    ax.legend(fontsize=6, loc="lower right", frameon=False)
    ax.tick_params(axis='both', labelsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def _mpl_fcf_bars(fcfs):
    fig, ax = plt.subplots(figsize=(4.0, 2.1), dpi=200)
    xs = np.arange(1, len(fcfs) + 1)
    ax.bar(xs, fcfs)
    ax.set_xlabel("Año")
    ax.set_ylabel("FCF (Gs.)")
    ax.set_xticks(xs)
    ax.tick_params(axis='both', labelsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def _mpl_hist(npv_vals, p5, p50, p95):
    fig, ax = plt.subplots(figsize=(4.0, 2.1), dpi=200)
    ax.hist(npv_vals, bins=35)
    for v, ls in [(p5, "--"), (p50, "-"), (p95, "--")]:
        ax.axvline(v, linestyle=ls, linewidth=1.2)
    ax.set_xlabel("VAN (Gs.)")
    ax.set_ylabel("Frecuencia")
    ax.tick_params(axis='both', labelsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_onepager_pdf(data: dict) -> bytes:
    if not (REPORTLAB_OK and MATPLOTLIB_OK):
        raise RuntimeError("PDF requiere reportlab + matplotlib instalados.")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    margin = 1.4 * cm
    x0, y0 = margin, H - margin

    # Header
    c.setFont("Helvetica-Bold", 12.5)
    c.drawString(x0, y0, "ONE-PAGER EJECUTIVO — EVALUACIÓN FINANCIERA (PYG)")
    c.setFont("Helvetica", 9.2)
    c.drawString(x0, y0 - 14, "Universidad San Ignacio de Loyola (USIL) — MBA — Proyectos de Inversión / Valuation")
    c.drawRightString(W - margin, y0, f"Fecha: {date.today().isoformat()}")
    c.drawString(x0, y0 - 28, f"Proyecto: {data['project']}  |  Responsable: {data['responsible']}")

    # KPI row
    y = y0 - 50
    c.setStrokeColorRGB(0.85, 0.85, 0.85)
    c.setLineWidth(0.6)
    c.line(margin, y, W - margin, y)

    y -= 18
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(x0, y, f"DICTAMEN: {data['verdict']}")
    c.setFont("Helvetica", 9.5)
    c.drawString(x0 + 115, y, data["rationale"])

    y -= 18
    c.setFont("Helvetica-Bold", 9.8)
    c.drawString(x0, y, "KPIs (determinístico)")
    c.setFont("Helvetica", 9.2)
    y -= 14
    c.drawString(x0, y, f"• VAN (base): {fmt_pyg(data['npv_base'])}")
    y -= 12
    c.drawString(x0, y, f"• TIR (base): {fmt_pct(data['irr_base']) if data['irr_base'] is not None else 'N/A'}")
    y -= 12
    c.drawString(x0, y, f"• Payback simple: {data['pb_simple']:.2f} años" if data['pb_simple'] is not None else "• Payback simple: N/A")
    y -= 12
    c.drawString(x0, y, f"• Payback descont.: {data['pb_disc']:.2f} años" if data['pb_disc'] is not None else "• Payback descont.: N/A")

    # Monte Carlo small block
    y_mc = y0 - 82
    x_mc = W * 0.52
    c.setFont("Helvetica-Bold", 9.8)
    c.drawString(x_mc, y_mc, "Monte Carlo — lectura ejecutiva")
    c.setFont("Helvetica", 9.2)
    y_mc -= 14
    c.drawString(x_mc, y_mc, f"Simulaciones: {data['sims']:,} | válidas: {data['valid_rate']:.1%} | P(VAN<0): {data['prob_neg']:.1%}")
    y_mc -= 12
    c.drawString(x_mc, y_mc, f"P5/P50/P95: {fmt_pyg(data['p5'])} / {fmt_pyg(data['p50'])} / {fmt_pyg(data['p95'])}")
    y_mc -= 12
    c.drawString(x_mc, y_mc, f"Media: {fmt_pyg(data['mean'])} | σ: {fmt_pyg(data['std'])} | CVaR5: {fmt_pyg(data['cvar5'])}")

    # Assumptions + flows + checklist
    y2 = y - 18
    c.setFont("Helvetica-Bold", 9.8)
    c.drawString(x0, y2, "Supuestos clave")
    c.setFont("Helvetica", 9.2)
    y2 -= 14
    c.drawString(x0, y2, f"• Horizonte: {data['n_years']} años + perpetuidad")
    y2 -= 12
    c.drawString(x0, y2, f"• g explícito: {data['g_exp']*100:.2f}% | g∞: {data['g_inf']*100:.2f}%")
    y2 -= 12
    c.drawString(x0, y2, f"• WACC: {data['wacc']*100:.2f}% (Ke {data['ke']*100:.2f}% | Kd {data['kd']*100:.2f}%)")
    y2 -= 12
    c.drawString(x0, y2, f"• CAPEX₀: {fmt_pyg(data['capex0'])} | FCF1: {fmt_pyg(data['fcf1'])}")

    y2 -= 16
    c.setFont("Helvetica-Bold", 9.8)
    c.drawString(x0, y2, "Flujos proyectados (FCF) — ciclo 1..N + TV (separado)")
    c.setFont("Helvetica", 9.1)
    y2 -= 14
    for i, v in enumerate(data["fcf_years"], start=1):
        if y2 < 9 * cm:  # avoid overflow; keep one pager
            break
        c.drawString(x0, y2, f"• Año {i}: {fmt_pyg(v)}")
        y2 -= 11
    c.drawString(x0, y2, f"• TV (en año {data['n_years']}): {fmt_pyg(data['tv'])}")
    y2 -= 16

    c.setFont("Helvetica-Bold", 9.8)
    c.drawString(x0, y2, "Checklist Comité (automático)")
    c.setFont("Helvetica", 9.2)
    y2 -= 14
    for line in data["checklist_lines"]:
        c.drawString(x0, y2, f"• {line}")
        y2 -= 11

    # Charts block
    # Place charts on bottom half: value bridge + hist + fcf bars
    chart_top = 9.2 * cm
    # Value bridge
    vb = _mpl_value_bridge(data["pv_fcf"], data["pv_tv"], data["capex0"])
    c.drawImage(ImageReader(vb), x0, chart_top, width=8.6*cm, height=3.7*cm, mask="auto")
    c.setFont("Helvetica", 8.5)
    c.drawString(x0, chart_top - 10, "Estructura del valor (PV FCF + PV TV − CAPEX₀)")

    # FCF bars
    fb = _mpl_fcf_bars(data["fcf_years"])
    c.drawImage(ImageReader(fb), x0, chart_top - 5.0*cm, width=8.6*cm, height=4.0*cm, mask="auto")
    c.setFont("Helvetica", 8.5)
    c.drawString(x0, chart_top - 5.0*cm - 10, "FCF por año (ciclo 1..N)")

    # Histogram
    hb = _mpl_hist(data["npv_valid"], data["p5"], data["p50"], data["p95"])
    c.drawImage(ImageReader(hb), x_mc, chart_top - 0.2*cm, width=8.6*cm, height=7.6*cm, mask="auto")
    c.setFont("Helvetica", 8.5)
    c.drawString(x_mc, chart_top - 10, "Distribución del VAN (Monte Carlo) con P5/P50/P95")

    # Footer
    c.setFont("Helvetica", 8.2)
    c.setFillColor(colors.grey)
    c.drawString(x0, margin * 0.9, "Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence.")
    c.drawRightString(W - margin, margin * 0.9, "ValuationSuite USIL (PYG)")
    c.save()

    buf.seek(0)
    return buf.read()


# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🧩 Identificación")
project = st.sidebar.text_input("Proyecto", value="Proyecto")
responsible = st.sidebar.text_input("Responsable", value="Docente: Jorge Rojas")

st.sidebar.divider()
st.sidebar.header("0) Inversión inicial")
capex0 = st.sidebar.number_input("CAPEX Año 0 (inversión inicial)", min_value=1.0, value=3_500_000_000.0, step=50_000_000.0)

st.sidebar.divider()
st.sidebar.header("1) CAPM / WACC (Nominal, D/E contable)")
rf = st.sidebar.number_input("Rf (%)", value=6.0, step=0.1) / 100
erp = st.sidebar.number_input("ERP (%)", value=5.5, step=0.1) / 100
crp = st.sidebar.number_input("CRP (%)", value=2.0, step=0.1) / 100
beta_u = st.sidebar.number_input("βU (desapalancada)", value=0.90, step=0.05)
tax_rate = st.sidebar.number_input("Impuesto (T) (%)", value=10.0, step=0.5) / 100

st.sidebar.divider()
st.sidebar.header("2) Estructura de capital (valores contables)")
debt = st.sidebar.number_input("Deuda (D)", min_value=0.0, value=2_000_000_000.0, step=50_000_000.0)
equity = st.sidebar.number_input("Capital propio (E)", min_value=1.0, value=3_000_000_000.0, step=50_000_000.0)
kd = st.sidebar.number_input("Costo de deuda Kd (%)", value=7.0, step=0.1) / 100

st.sidebar.divider()
st.sidebar.header("3A) Contable Año 1 (para calcular FCF Año 1)")
sales_y1 = st.sidebar.number_input("Ventas Año 1", min_value=0.0, value=6_000_000_000.0, step=50_000_000.0)
var_cost_y1 = st.sidebar.number_input("Costos variables Año 1", min_value=0.0, value=3_000_000_000.0, step=50_000_000.0)
fixed_cost_y1 = st.sidebar.number_input("Costos fijos Año 1", min_value=0.0, value=1_500_000_000.0, step=50_000_000.0)
dep_y1 = st.sidebar.number_input("Depreciación Año 1 (no caja)", min_value=0.0, value=250_000_000.0, step=10_000_000.0)
capex_y1 = st.sidebar.number_input("CAPEX Año 1 (mantenimiento/crecimiento)", min_value=0.0, value=250_000_000.0, step=10_000_000.0)

st.sidebar.markdown("**Δ Capital de trabajo (Año 1) — componentes (simplificado)**")
d_ar = st.sidebar.number_input("Δ Cuentas por cobrar (AR)", value=120_000_000.0, step=10_000_000.0)
d_inv = st.sidebar.number_input("Δ Inventarios (INV)", value=60_000_000.0, step=10_000_000.0)
d_ap = st.sidebar.number_input("Δ Cuentas por pagar (AP)", value=40_000_000.0, step=10_000_000.0)

st.sidebar.divider()
st.sidebar.header("3B) Flujos (Ciclo 1..N) + g perpetuidad")
n_years = int(st.sidebar.slider("Años de proyección (N)", min_value=3, max_value=10, value=5, step=1))
g_exp = st.sidebar.number_input("g explícito (%) (crecimiento ciclo 1..N)", value=5.0, step=0.25) / 100
g_inf = st.sidebar.number_input("g a perpetuidad (%)", value=2.0, step=0.1) / 100

st.sidebar.divider()
st.sidebar.header("4) Monte Carlo (incertidumbre razonable)")
sims = int(st.sidebar.slider("Simulaciones", min_value=5000, max_value=60000, value=DEFAULT_SIMS, step=1000))
st.sidebar.caption("Se asume RNG interno (sin semilla expuesta).")

st.sidebar.markdown("**Rangos triangulares (mín=adverso, base=más probable, máx=favorable)**")
g_min = st.sidebar.number_input("g mín (%)", value=max(-20.0, (g_exp*100) - 4.0), step=0.25) / 100
g_mode = st.sidebar.number_input("g base (%)", value=g_exp*100, step=0.25) / 100
g_max = st.sidebar.number_input("g máx (%)", value=(g_exp*100) + 4.0, step=0.25) / 100

wacc_pm = st.sidebar.number_input("WACC rango ± (%)", value=15.0, step=1.0) / 100

capex_min = st.sidebar.number_input("CAPEX mín (favorable)", min_value=0.0, value=capex0 * 0.85, step=10_000_000.0)
capex_mode = st.sidebar.number_input("CAPEX base", min_value=0.0, value=capex0, step=10_000_000.0)
capex_max = st.sidebar.number_input("CAPEX máx (adverso)", min_value=0.0, value=capex0 * 1.20, step=10_000_000.0)

fcf_mult_min = st.sidebar.number_input("Shock FCF1 mín (adverso)", value=0.85, step=0.01)
fcf_mult_mode = st.sidebar.number_input("Shock FCF1 base", value=1.00, step=0.01)
fcf_mult_max = st.sidebar.number_input("Shock FCF1 máx (favorable)", value=1.15, step=0.01)


# -----------------------------
# Core calculations
# -----------------------------
D = safe_float(debt, 0.0)
E = safe_float(equity, 1.0)
V = max(D + E, 1.0)

beta_l = beta_u * (1 + (1 - tax_rate) * (D / E))
ke = rf + beta_l * erp + crp
wacc = (E / V) * ke + (D / V) * kd * (1 - tax_rate)

# Accounting to FCF
ebit = sales_y1 - var_cost_y1 - fixed_cost_y1 - dep_y1
nopat = ebit * (1 - tax_rate)
d_wc = d_ar + d_inv - d_ap
fcf1 = nopat + dep_y1 - capex_y1 - d_wc

# Flows + TV
years = np.arange(1, n_years + 1)
fcf_years = np.array([fcf1 * ((1 + g_exp) ** (t - 1)) for t in years], dtype=float)

valid_tv = wacc > (g_inf + MIN_SPREAD)
tv = np.nan
npv_base = np.nan
pv_fcf = np.nan
pv_tv = np.nan

if valid_tv:
    tv = (fcf_years[-1] * (1 + g_inf)) / (wacc - g_inf)
    disc = (1 + wacc) ** years
    pv_fcf = float(np.sum(fcf_years / disc))
    pv_tv = float(tv / ((1 + wacc) ** n_years))
    npv_base = pv_fcf + pv_tv - capex0

cashflows = [-capex0] + list(fcf_years)
cashflows_tv = cashflows[:-1] + [cashflows[-1] + (tv if np.isfinite(tv) else 0.0)]
irr_base = safe_irr(cashflows_tv) if valid_tv else None

pb_simple = payback_simple(cashflows_tv)
pb_disc = payback_discounted(cashflows_tv, wacc) if valid_tv else None

# Monte Carlo
npv_sims, valid_mask = run_monte_carlo(
    sims=sims,
    fcf1=float(fcf1),
    n_years=int(n_years),
    g_inf=float(g_inf),
    wacc_base=float(wacc),
    wacc_pm=float(wacc_pm),
    g_min=float(g_min),
    g_mode=float(g_mode),
    g_max=float(g_max),
    capex_min=float(capex_min),
    capex_mode=float(capex_mode),
    capex_max=float(capex_max),
    fcf_mult_min=float(fcf_mult_min),
    fcf_mult_mode=float(fcf_mult_mode),
    fcf_mult_max=float(fcf_mult_max),
)

npv_valid = npv_sims[np.isfinite(npv_sims)]
valid_rate = float(np.mean(np.isfinite(npv_sims))) if npv_sims.size else 0.0

if npv_valid.size:
    p5, p50, p95 = np.percentile(npv_valid, [5, 50, 95])
    mean = float(np.mean(npv_valid))
    std = float(np.std(npv_valid, ddof=1)) if npv_valid.size > 1 else 0.0
    prob_neg = float(np.mean(npv_valid < 0))
    var5 = float(p5)
    cvar5 = float(np.mean(npv_valid[npv_valid <= p5])) if np.any(npv_valid <= p5) else float(p5)
else:
    p5 = p50 = p95 = mean = std = prob_neg = var5 = cvar5 = np.nan

# Committee checklist (automatic defaults)
van_ok = bool(np.isfinite(npv_base) and npv_base > 0)
pneg_ok = bool(np.isfinite(prob_neg) and prob_neg <= 0.20)
p50_ok = bool(np.isfinite(p50) and p50 > 0)
p5_ok = bool(np.isfinite(p5) and p5 > 0)

verdict, verdict_class = verdict_from_checks(van_ok, pneg_ok, p50_ok, p5_ok)

rationale_map = {
    "ok": "El proyecto cumple criterios conservadores: creación de valor y downside controlado bajo incertidumbre razonable.",
    "warn": "El caso es prometedor, pero requiere reforzar supuestos/mitigaciones para reducir downside antes de aprobar.",
    "bad": "El perfil riesgo–retorno no cumple mínimos; se recomienda rediseño del caso o mitigaciones fuertes.",
}
rationale = rationale_map[verdict_class]

checklist_lines = [
    f"{'✅' if van_ok else '⛔'} VAN base > 0",
    f"{'✅' if pneg_ok else '⛔'} P(VAN<0) ≤ 20%",
    f"{'✅' if p50_ok else '⛔'} P50(VAN) > 0",
    f"{'✅' if p5_ok else '⛔'} P5(VAN) > 0",
]

# -----------------------------
# Header
# -----------------------------
st.markdown("## ONE-PAGER EJECUTIVO — EVALUACIÓN FINANCIERA (PYG)")
st.markdown(
    f"<div class='small-muted'>Universidad San Ignacio de Loyola (USIL) — MBA — Proyectos de Inversión / Valuation</div>",
    unsafe_allow_html=True,
)

# Keep dates clean (no overlap): explicit 2-col layout
h1, h2 = st.columns([0.70, 0.30])
with h1:
    st.markdown(f"<div class='small-muted'>Proyecto: <b>{project}</b> &nbsp;|&nbsp; Responsable: <b>{responsible}</b></div>", unsafe_allow_html=True)
with h2:
    st.markdown(f"<div class='small-muted' style='text-align:right;'>Fecha: <b>{date.today().isoformat()}</b></div>", unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# -----------------------------
# KPI cards
# -----------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)

def card(col, title, big, sub):
    col.markdown(
        f"""
<div class="card">
  <h4>{title}</h4>
  <div class="big">{big}</div>
  <div class="sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )

card(k1, "VAN (base)", fmt_pyg(npv_base) if np.isfinite(npv_base) else "—", "Determinístico")
card(k2, "TIR (base)", fmt_pct(irr_base), "Determinístico")
card(k3, "Payback", f"{pb_simple:.2f} a" if pb_simple is not None else "—", "Simple")
card(k4, "Payback (desc.)", f"{pb_disc:.2f} a" if pb_disc is not None else "—", "Descontado")
card(k5, "P(VAN<0)", f"{prob_neg*100:.1f}%" if np.isfinite(prob_neg) else "—", "Monte Carlo")
card(k6, "P50 (VAN)", fmt_pyg(p50) if np.isfinite(p50) else "—", "Monte Carlo")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# -----------------------------
# Main 2x2 grid
# -----------------------------
c_left, c_right = st.columns([0.52, 0.48])

# Decision & assumptions
with c_left:
    pill = f"<span class='pill {verdict_class}'>DICTAMEN: <b>{verdict}</b></span>"
    st.markdown(
        f"""
<div class="card">
  <h3 style="margin:0 0 8px 0; font-size:1.05rem;">Decisión & supuestos</h3>
  {pill}
  <div class="small-muted" style="margin-top:8px;">{rationale}</div>
  <div class="hr"></div>
  <div class="small-muted">
    • Horizonte: <b>{n_years} años</b> + perpetuidad<br/>
    • g explícito: <b>{g_exp*100:.2f}%</b> &nbsp;|&nbsp; g∞: <b>{g_inf*100:.2f}%</b><br/>
    • WACC: <b>{wacc*100:.2f}%</b> (Ke {ke*100:.2f}% | Kd {kd*100:.2f}%)<br/>
    • CAPEX₀: <b>{fmt_pyg(capex0)}</b> &nbsp;|&nbsp; FCF₁ (calculado): <b>{fmt_pyg(fcf1)}</b><br/>
    • Simulaciones: <b>{sims:,}</b> (válidas: {valid_rate:.1%})
  </div>
  <div class="hr"></div>
  <div class="small-muted"><b>Checklist Comité (automático)</b><br/>
    {checklist_lines[0]}<br/>
    {checklist_lines[1]}<br/>
    {checklist_lines[2]}<br/>
    {checklist_lines[3]}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# Value structure chart
with c_right:
    st.markdown(
        """
<div class="card">
  <h3 style="margin:0 0 6px 0; font-size:1.05rem;">Estructura del valor</h3>
  <div class="small-muted">VAN = PV(FCF ciclo 1..N) + PV(TV) − CAPEX₀ (TV separado)</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if np.isfinite(pv_fcf) and np.isfinite(pv_tv):
        fig = go.Figure()
        fig.add_trace(go.Bar(name="PV FCF", y=["VAN"], x=[pv_fcf], orientation="h"))
        fig.add_trace(go.Bar(name="PV TV", y=["VAN"], x=[pv_tv], orientation="h"))
        fig.add_trace(go.Bar(name="CAPEX₀", y=["VAN"], x=[-capex0], orientation="h"))
        fig.update_layout(
            barmode="relative",
            height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,.85)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"<div class='small-muted' style='text-align:right; margin-top:-6px;'><b>VAN (base): {fmt_pyg(npv_base)}</b></div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("WACC debe ser mayor que g∞ + 0.5% para calcular TV y VAN. Ajusta supuestos.")

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Bottom charts row: FCF + Monte Carlo hist
b1, b2 = st.columns([0.52, 0.48])

with b1:
    st.markdown(
        """
<div class="card">
  <h3 style="margin:0 0 6px 0; font-size:1.05rem;">Flujos proyectados (ciclo 1..N) + TV</h3>
  <div class="small-muted">FCF por ciclo. El valor terminal (TV) se muestra separado.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=[str(i) for i in years], y=fcf_years, name="FCF"))
    fig2.update_layout(
        height=270,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,.85)"),
    )
    st.plotly_chart(fig2, use_container_width=True)
    if np.isfinite(tv):
        st.markdown(f"<div class='small-muted'>TV (en ciclo {n_years}): <b>{fmt_pyg(tv)}</b></div>", unsafe_allow_html=True)

with b2:
    st.markdown(
        """
<div class="card">
  <h3 style="margin:0 0 6px 0; font-size:1.05rem;">Riesgo (Monte Carlo) — distribución del VAN</h3>
  <div class="small-muted">Histograma con P5/P50/P95. Lectura ejecutiva ampliada debajo.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if npv_valid.size:
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=npv_valid, nbinsx=40, name="VAN"))
        for v, nm in [(p5, "P5"), (p50, "P50"), (p95, "P95")]:
            fig3.add_vline(x=v, line_dash="dash" if nm != "P50" else "solid", line_width=2)
        fig3.update_layout(
            height=270,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,.85)"),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Expanded Monte Carlo explanation (keeps layout stable)
        st.markdown(
            f"""
<div class="small-muted">
<b>Lectura ejecutiva (Monte Carlo):</b><br/>
• <b>P(VAN&lt;0)</b> = <b>{prob_neg*100:.1f}%</b>: proporción de escenarios que destruyen valor.<br/>
• <b>P5</b> = <b>{fmt_pyg(p5)}</b>: “downside plausible” (1 de cada 20 escenarios peores).<br/>
• <b>P50</b> = <b>{fmt_pyg(p50)}</b>: escenario central (mediana).<br/>
• <b>P95</b> = <b>{fmt_pyg(p95)}</b>: upside plausible (1 de cada 20 escenarios mejores).<br/>
• <b>Media</b> = {fmt_pyg(mean)} | <b>σ</b> = {fmt_pyg(std)} | <b>CVaR5</b> = {fmt_pyg(cvar5)} (promedio del 5% peor).<br/>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.warning("No hay simulaciones válidas (revisa WACC vs g∞ y rangos).")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# Accounting bridge (compact but clear)
st.markdown(
    """
<div class="card">
  <h3 style="margin:0 0 6px 0; font-size:1.05rem;">Puente contable → FCF (Año 1)</h3>
  <div class="small-muted">Pensado para estudiantes sin fondo contable: EBIT → NOPAT → ajustes no caja → CAPEX → ΔCT.</div>
</div>
""",
    unsafe_allow_html=True,
)
bcol1, bcol2, bcol3, bcol4 = st.columns(4)
card(bcol1, "EBIT", fmt_pyg(ebit), "Ventas − CV − CF − Dep.")
card(bcol2, "NOPAT", fmt_pyg(nopat), "EBIT × (1−T)")
card(bcol3, "ΔCT (AR+INV−AP)", fmt_pyg(d_wc), "Uso de caja (si +)")
card(bcol4, "FCF Año 1", fmt_pyg(fcf1), "NOPAT + Dep − CAPEX − ΔCT")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<div class='small-muted'>Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence. · <b>ValuationSuite USIL (PYG)</b></div>",
    unsafe_allow_html=True,
)

# -----------------------------
# Exports
# -----------------------------
st.markdown("### Export")

# Build onepager data dict (avoid dataclass ordering issues on Streamlit Cloud)
onepager = {
    "project": project,
    "responsible": responsible,
    "verdict": verdict,
    "rationale": rationale,
    "capex0": float(capex0),
    "n_years": int(n_years),
    "g_exp": float(g_exp),
    "g_inf": float(g_inf),
    "wacc": float(wacc),
    "ke": float(ke),
    "kd": float(kd),
    "fcf1": float(fcf1),
    "npv_base": float(npv_base) if np.isfinite(npv_base) else float("nan"),
    "irr_base": float(irr_base) if irr_base is not None else None,
    "pb_simple": pb_simple,
    "pb_disc": pb_disc,
    "tv": float(tv) if np.isfinite(tv) else float("nan"),
    "fcf_years": [float(x) for x in fcf_years],
    "pv_fcf": float(pv_fcf) if np.isfinite(pv_fcf) else float("nan"),
    "pv_tv": float(pv_tv) if np.isfinite(pv_tv) else float("nan"),
    "sims": int(sims),
    "valid_rate": float(valid_rate),
    "prob_neg": float(prob_neg) if np.isfinite(prob_neg) else float("nan"),
    "p5": float(p5) if np.isfinite(p5) else float("nan"),
    "p50": float(p50) if np.isfinite(p50) else float("nan"),
    "p95": float(p95) if np.isfinite(p95) else float("nan"),
    "mean": float(mean) if np.isfinite(mean) else float("nan"),
    "std": float(std) if np.isfinite(std) else float("nan"),
    "cvar5": float(cvar5) if np.isfinite(cvar5) else float("nan"),
    "checklist_lines": checklist_lines,
    "npv_valid": npv_valid.tolist(),
}

# TXT export (always available)
txt_lines = []
txt_lines.append("ONE-PAGER EJECUTIVO — EVALUACIÓN FINANCIERA (PYG)")
txt_lines.append("Universidad San Ignacio de Loyola (USIL) — MBA — Proyectos de Inversión / Valuation")
txt_lines.append(f"Fecha: {date.today().isoformat()}")
txt_lines.append(f"Proyecto: {project} | Responsable: {responsible}")
txt_lines.append("")
txt_lines.append(f"DICTAMEN: {verdict}")
txt_lines.append(rationale)
txt_lines.append("")
txt_lines.append("KPIs (determinístico)")
txt_lines.append(f"- VAN (base): {fmt_pyg(npv_base)}")
txt_lines.append(f"- TIR (base): {fmt_pct(irr_base)}")
txt_lines.append(f"- Payback simple: {pb_simple:.2f} años" if pb_simple is not None else "- Payback simple: N/A")
txt_lines.append(f"- Payback descontado: {pb_disc:.2f} años" if pb_disc is not None else "- Payback descontado: N/A")
txt_lines.append("")
txt_lines.append("Flujos proyectados (FCF) — ciclo 1..N")
for i, v in enumerate(fcf_years, start=1):
    txt_lines.append(f"- Año {i}: {fmt_pyg(v)}")
txt_lines.append(f"- TV (en año {n_years}): {fmt_pyg(tv)}")
txt_lines.append("")
txt_lines.append("Monte Carlo")
txt_lines.append(f"- Simulaciones: {sims:,} | válidas: {valid_rate:.1%}")
txt_lines.append(f"- P(VAN<0): {prob_neg*100:.1f}%")
txt_lines.append(f"- P5/P50/P95: {fmt_pyg(p5)} / {fmt_pyg(p50)} / {fmt_pyg(p95)}")
txt_lines.append(f"- Media: {fmt_pyg(mean)} | σ: {fmt_pyg(std)} | CVaR5: {fmt_pyg(cvar5)}")
txt_lines.append("")
txt_lines.append("Checklist Comité (automático)")
for l in checklist_lines:
    txt_lines.append(f"- {l}")
txt_lines.append("")
txt_lines.append("Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence.")
txt_bytes = ("\n".join(txt_lines)).encode("utf-8")

st.download_button("⬇️ Descargar One-Pager (TXT)", data=txt_bytes, file_name="one_pager_usil.txt", mime="text/plain")

# PDF export (if deps)
if REPORTLAB_OK and MATPLOTLIB_OK:
    try:
        pdf_bytes = generate_onepager_pdf(onepager)
        st.download_button("⬇️ Descargar One-Pager (PDF)", data=pdf_bytes, file_name="one_pager_usil.pdf", mime="application/pdf")
    except Exception as e:
        st.warning(f"No se pudo generar PDF: {e}")
else:
    st.info("Para exportar PDF, asegúrate de tener `reportlab` y `matplotlib` en requirements.txt.")
