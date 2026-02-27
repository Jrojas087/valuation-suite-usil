# app.py
# ValuationSuite USIL (PYG) — Streamlit One‑Pager Dashboard + PDF Premium
# Autor: (generado con ayuda de ChatGPT)
# ------------------------------------------------------------
# Requisitos: ver requirements.txt (NO usa matplotlib; PDF 100% reportlab)

import io
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy_financial as npf
import plotly.express as px
import streamlit as st

import report_usil as rep


# ============================================================
# Config UI
# ============================================================
st.set_page_config(
    page_title="ValuationSuite USIL — Evaluación Financiera (PYG)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Estilo (premium dark dashboard)
# ============================================================
DARK_CSS = """
<style>
:root{
  --bg0:#050914; --bg1:#071026; --card:#0b1733; --card2:#0d1b3d;
  --line:rgba(255,255,255,.08); --text:#eaf1ff; --muted:rgba(234,241,255,.72);
  --accent:#66a9ff; --good:#27d17c; --warn:#ffcc66; --bad:#ff5d5d;
}
html, body, [class*="stApp"] { background: radial-gradient(1200px 700px at 20% 0%, #0a1736 0%, var(--bg0) 55%, #04070f 100%) !important; color: var(--text) !important; }
h1,h2,h3,h4,h5,h6, p, div, span, label { color: var(--text); }
[data-testid="stSidebar"]{ background: linear-gradient(180deg, var(--bg1), #050914) !important; border-right: 1px solid var(--line); }
[data-testid="stMetric"]{ background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02)); border: 1px solid var(--line); padding: 14px 14px; border-radius: 14px; }
.block-container { padding-top: 1.2rem; }
hr { border-color: var(--line); }
.card {
  background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
}
.card h3 { margin: 0 0 8px 0; font-size: 1.05rem; }
.small { color: var(--muted); font-size: .88rem; }
.pill {
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(102,169,255,.10);
  font-weight: 700; letter-spacing: .02em;
}
.pill.good{ background: rgba(39,209,124,.12); }
.pill.warn{ background: rgba(255,204,102,.12); }
.pill.bad{ background: rgba(255,93,93,.12); }
.kpiSub { color: var(--muted); font-size: .82rem; margin-top: -2px; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

st.title("📊 ValuationSuite USIL — One‑Pager Ejecutivo (PYG)")
st.caption("DCF + lectura probabilística (Monte Carlo). Modelo didáctico para toma de decisiones. 🇵🇾")


# ============================================================
# Utilidades
# ============================================================
MIN_SPREAD = 0.005  # wacc > g_inf + spread

def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def fmt_pyg(x: float) -> str:
    # Gs. con separador de miles, 0 decimales (PYG)
    return "Gs. {:,.0f}".format(float(x)).replace(",", ".")

def safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

def safe_irr(cashflows):
    try:
        irr = float(npf.irr(cashflows))
        if np.isnan(irr) or np.isinf(irr):
            return None
        # filtro conservador
        if irr < -0.99 or irr > 2.0:
            return None
        return irr
    except Exception:
        return None

def payback_simple(capex0: float, fcfs: np.ndarray) -> float | None:
    cum = -capex0
    for i, f in enumerate(fcfs, start=1):
        prev = cum
        cum += f
        if cum >= 0:
            if f == 0:
                return float(i)
            frac = (0 - prev) / f
            return float((i-1) + frac)
    return None

def payback_discounted(capex0: float, fcfs: np.ndarray, wacc: float) -> float | None:
    cum = -capex0
    for i, f in enumerate(fcfs, start=1):
        pv = f / ((1+wacc)**i)
        prev = cum
        cum += pv
        if cum >= 0:
            if pv == 0:
                return float(i)
            frac = (0 - prev) / pv
            return float((i-1) + frac)
    return None

def committee_check(npv_base: float, prob_neg: float, p50: float, p5: float) -> tuple[str, list[tuple[str,bool]]]:
    checks = [
        ("VAN base > 0", npv_base > 0),
        ("P(VAN<0) ≤ 20%", prob_neg <= 0.20),
        ("P50(VAN) > 0", p50 > 0),
        ("P5(VAN) > 0", p5 > 0),
    ]
    ok = sum(b for _, b in checks)
    if ok == len(checks):
        return "APROBADO", checks
    if ok == 0:
        return "RECHAZADO", checks
    return "OBSERVADO", checks


# ============================================================
# Monte Carlo (triangular) — sin semilla expuesta
# ============================================================
@st.cache_data(show_spinner=False)
def run_monte_carlo(
    sims: int,
    fcf_y1: float,
    n_years: int,
    g_inf: float,
    # tri g explícito
    g_min: float, g_mode: float, g_max: float,
    # tri wacc (desde ±range alrededor del wacc base)
    w_min: float, w_mode: float, w_max: float,
    # tri capex0
    capex_min: float, capex_mode: float, capex_max: float,
    # tri shock FCF1
    fcf_mult_min: float, fcf_mult_mode: float, fcf_mult_max: float,
):
    rng = np.random.default_rng()

    g_s = rng.triangular(g_min, g_mode, g_max, sims)
    w_s = rng.triangular(w_min, w_mode, w_max, sims)
    capex_s = rng.triangular(capex_min, capex_mode, capex_max, sims)
    mult_s = rng.triangular(fcf_mult_min, fcf_mult_mode, fcf_mult_max, sims)

    yrs = np.arange(1, n_years + 1)
    fcf1_s = fcf_y1 * mult_s
    fcf_paths = fcf1_s[:, None] * (1.0 + g_s)[:, None] ** (yrs[None, :] - 1)

    valid = w_s > (g_inf + MIN_SPREAD)
    npv_s = np.full(sims, np.nan)

    idx = np.where(valid)[0]
    if idx.size:
        fcf_valid = fcf_paths[idx, :]
        w_valid = w_s[idx]
        capex_valid = capex_s[idx]

        tv_valid = (fcf_valid[:, -1] * (1.0 + g_inf)) / (w_valid - g_inf)
        fcf_valid[:, -1] += tv_valid

        disc = (1.0 + w_valid)[:, None] ** (yrs[None, :])
        pv = np.sum(fcf_valid / disc, axis=1)
        npv_s[idx] = pv - capex_valid

    return npv_s, g_s, w_s, capex_s, mult_s, idx


# ============================================================
# Sidebar inputs
# ============================================================
st.sidebar.header("🧩 Identificación")
project = st.sidebar.text_input("Proyecto", "Proyecto")
responsible = st.sidebar.text_input("Responsable", "Docente: Jorge Rojas")

st.sidebar.divider()
st.sidebar.header("0) Inversión inicial")
capex0 = st.sidebar.number_input("CAPEX Año 0 (inversión inicial)", value=3_500_000_000.0, step=50_000_000.0, min_value=1.0)

st.sidebar.divider()
st.sidebar.header("1) CAPM / WACC (Nominal, D/E contable)")
rf = st.sidebar.number_input("Rf (%)", value=4.5, step=0.1) / 100.0
erp = st.sidebar.number_input("ERP (%)", value=5.5, step=0.1) / 100.0
crp = st.sidebar.number_input("CRP (%)", value=2.0, step=0.1) / 100.0
beta_u = st.sidebar.number_input("βU (desapalancada)", value=0.90, step=0.05)
tax_rate = st.sidebar.number_input("Impuesto (T) (%)", value=10.0, step=0.5) / 100.0

st.sidebar.divider()
st.sidebar.header("2) Estructura de capital (valores contables)")
debt = st.sidebar.number_input("Deuda (D)", value=2_000_000_000.0, step=50_000_000.0, min_value=0.0)
equity = st.sidebar.number_input("Capital propio (E)", value=3_000_000_000.0, step=50_000_000.0, min_value=1.0)
kd = st.sidebar.number_input("Costo de deuda Kd (%)", value=7.0, step=0.25) / 100.0

# 3A) Puente contable → FCF1
st.sidebar.divider()
st.sidebar.header("3A) Contable Año 1 (para calcular FCF Año 1)")
sales_y1 = st.sidebar.number_input("Ventas Año 1", value=10_000_000_000.0, step=100_000_000.0, min_value=0.0)
cvar_y1 = st.sidebar.number_input("Costos variables Año 1", value=6_000_000_000.0, step=100_000_000.0, min_value=0.0)
cfix_y1 = st.sidebar.number_input("Costos fijos Año 1", value=2_750_000_000.0, step=50_000_000.0, min_value=0.0)
dep_y1 = st.sidebar.number_input("Depreciación Año 1 (no caja)", value=1_500_000_000.0, step=50_000_000.0, min_value=0.0)
capex_y1 = st.sidebar.number_input("CAPEX Año 1 (mantenimiento/crecimiento)", value=250_000_000.0, step=25_000_000.0, min_value=0.0)

st.sidebar.markdown("**Δ Capital de trabajo (Año 1) — componentes (simplificado)**")
d_ar = st.sidebar.number_input("Δ Cuentas por cobrar (AR)", value=90_000_000.0, step=10_000_000.0)
d_inv = st.sidebar.number_input("Δ Inventarios (INV)", value=80_000_000.0, step=10_000_000.0)
d_ap = st.sidebar.number_input("Δ Cuentas por pagar (AP)", value=30_000_000.0, step=10_000_000.0)

# 3B) Proyección explícita + perpetuidad
st.sidebar.divider()
st.sidebar.header("3B) Flujos (Ciclo 1..N) + g perpetuidad")
n_years = int(st.sidebar.slider("Años de proyección (N)", min_value=3, max_value=10, value=5))
g_exp = st.sidebar.number_input("g explícito (%) (crecimiento ciclo 1..N)", value=5.0, step=0.25) / 100.0
g_inf = st.sidebar.number_input("g a perpetuidad (%)", value=2.0, step=0.10) / 100.0

# Monte Carlo
st.sidebar.divider()
st.sidebar.header("4) Monte Carlo (incertidumbre razonable)")
sims = int(st.sidebar.slider("Simulaciones", min_value=5000, max_value=60000, value=15000, step=1000))
st.sidebar.caption("Se asume RNG interno (sin semilla expuesta). Rangos triangulares.")

g_min = st.sidebar.number_input("g mín (%)", value=2.0, step=0.25) / 100.0
g_mode = st.sidebar.number_input("g base (%)", value=5.0, step=0.25) / 100.0
g_max = st.sidebar.number_input("g máx (%)", value=8.0, step=0.25) / 100.0

wacc_range = st.sidebar.number_input("WACC rango ± (%)", value=2.0, step=0.25) / 100.0

capex_min = st.sidebar.number_input("CAPEX mín (favorable)", value=max(capex0*0.90, 1.0), step=50_000_000.0)
capex_mode = st.sidebar.number_input("CAPEX base", value=capex0, step=50_000_000.0)
capex_max = st.sidebar.number_input("CAPEX máx (adverso)", value=capex0*1.10, step=50_000_000.0)

mult_min = st.sidebar.number_input("Shock FCF1 mín (adverso)", value=0.85, step=0.01)
mult_mode = st.sidebar.number_input("Shock FCF1 base", value=1.00, step=0.01)
mult_max = st.sidebar.number_input("Shock FCF1 máx (favorable)", value=1.15, step=0.01)


# ============================================================
# Cálculos base: WACC, FCF1, DCF
# ============================================================
# D/E contable
D = float(debt)
E = float(equity)
V = max(D + E, 1e-9)
wD = D / V
wE = E / V

# Beta apalancada por D/E contable
beta_l = beta_u * (1.0 + (1.0 - tax_rate) * (D / max(E, 1e-9)))

ke = rf + beta_l * erp + crp
wacc = wE * ke + wD * kd * (1.0 - tax_rate)

# Puente contable a FCF1
ebit = sales_y1 - cvar_y1 - cfix_y1 - dep_y1
nopat = ebit * (1.0 - tax_rate)
delta_wc = (d_ar + d_inv - d_ap)

fcf_y1 = nopat + dep_y1 - capex_y1 - delta_wc  # FCF Año 1 calculado

# FCF ciclo 1..N
years = np.arange(1, n_years + 1)
fcf_series = fcf_y1 * (1.0 + g_exp) ** (years - 1)

# TV y VAN determinístico
valid_tv = (wacc > (g_inf + MIN_SPREAD))
if valid_tv:
    tv = (fcf_series[-1] * (1.0 + g_inf)) / (wacc - g_inf)
else:
    tv = np.nan

pv_fcf = np.sum(fcf_series / (1.0 + wacc) ** years) if valid_tv else np.nan
pv_tv = (tv / (1.0 + wacc) ** n_years) if valid_tv else np.nan
npv_base = (pv_fcf + pv_tv - capex0) if valid_tv else np.nan

# Cashflows para IRR (CAPEX0 + FCFs + TV)
cashflows = [-capex0] + list(fcf_series)
cashflows[-1] = cashflows[-1] + (tv if np.isfinite(tv) else 0.0)
irr_base = safe_irr(cashflows)

pb_simple = payback_simple(capex0, fcf_series)
pb_disc = payback_discounted(capex0, fcf_series, wacc) if np.isfinite(wacc) else None


# ============================================================
# Monte Carlo
# ============================================================
w_min = max(wacc - wacc_range, 0.0)
w_mode = max(wacc, 0.0)
w_max = max(wacc + wacc_range, 0.0)

npv_s, g_s, w_s, capex_s, mult_s, valid_idx = run_monte_carlo(
    sims=sims,
    fcf_y1=float(fcf_y1),
    n_years=n_years,
    g_inf=float(g_inf),
    g_min=float(g_min), g_mode=float(g_mode), g_max=float(g_max),
    w_min=float(w_min), w_mode=float(w_mode), w_max=float(w_max),
    capex_min=float(capex_min), capex_mode=float(capex_mode), capex_max=float(capex_max),
    fcf_mult_min=float(mult_min), fcf_mult_mode=float(mult_mode), fcf_mult_max=float(mult_max),
)

valid_rate = float(np.isfinite(npv_s).mean())
npv_valid = npv_s[np.isfinite(npv_s)]
if npv_valid.size:
    prob_neg = float((npv_valid < 0).mean())
    p5, p50, p95 = np.percentile(npv_valid, [5, 50, 95])
    mean = float(np.mean(npv_valid))
    std = float(np.std(npv_valid))
    var5 = float(np.percentile(npv_valid, 5))
    cvar5 = float(np.mean(npv_valid[npv_valid <= var5])) if np.any(npv_valid <= var5) else var5
else:
    prob_neg = float("nan")
    p5 = p50 = p95 = mean = std = cvar5 = float("nan")


# ============================================================
# Dictamen + narrativa
# ============================================================
if not np.isfinite(npv_base):
    verdict = "OBSERVADO"
    checks = [("Consistencia TV (WACC > g∞ + spread)", False)]
    rationale = "La consistencia del valor terminal no se cumple (WACC debe ser mayor que g∞ + spread). Ajustar supuestos."
else:
    verdict, checks = committee_check(float(npv_base), float(prob_neg), float(p50), float(p5))
    if verdict == "APROBADO":
        rationale = ("El proyecto satisface criterios conservadores: creación de valor y downside controlado "
                     "bajo incertidumbre razonable.")
    elif verdict == "RECHAZADO":
        rationale = ("El proyecto no cumple criterios mínimos. Se recomienda rediseñar supuestos clave o estructura "
                     "de inversión antes de avanzar.")
    else:
        rationale = ("El proyecto muestra potencial, pero requiere reforzar supuestos críticos y mitigaciones "
                     "antes de aprobación final.")


# ============================================================
# Dashboard (One‑Pager visual)
# ============================================================
top = st.container()
with top:
    left, right = st.columns([0.75, 0.25], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="card">
              <div style="display:flex; justify-content:space-between; align-items:flex-end; gap:14px;">
                <div>
                  <div style="font-size:1.05rem; font-weight:800; letter-spacing:.02em;">ONE‑PAGER EJECUTIVO — EVALUACIÓN FINANCIERA (PYG)</div>
                  <div class="small">Universidad San Ignacio de Loyola (USIL) — MBA — Proyectos de Inversión / Valuation</div>
                  <div class="small">Proyecto: <b>{project}</b> &nbsp;|&nbsp; Responsable: <b>{responsible}</b></div>
                </div>
                <div class="small" style="text-align:right;">Fecha: <b>{date.today().isoformat()}</b></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        pill_class = "good" if verdict=="APROBADO" else ("bad" if verdict=="RECHAZADO" else "warn")
        st.markdown(
            f"""
            <div class="card" style="text-align:center;">
              <div class="small">Dictamen</div>
              <div class="pill {pill_class}" style="font-size:1.05rem;">{verdict}</div>
              <div class="small" style="margin-top:8px;">Checklist automático + Monte Carlo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")

c1, c2, c3, c4, c5, c6 = st.columns(6, gap="medium")
c1.metric("VAN (base)", fmt_pyg(npv_base) if np.isfinite(npv_base) else "—", "Determinístico")
c2.metric("TIR (base)", fmt_pct(irr_base) if irr_base is not None else "N/A", "Determinístico")
c3.metric("Payback", f"{pb_simple:.2f} años" if pb_simple is not None else "N/A", "Simple")
c4.metric("Payback (desc.)", f"{pb_disc:.2f} años" if pb_disc is not None else "N/A", "Descontado")
c5.metric("P(VAN<0)", f"{prob_neg*100:.1f}%" if np.isfinite(prob_neg) else "—", "Monte Carlo")
c6.metric("P50 (VAN)", fmt_pyg(p50) if np.isfinite(p50) else "—", "Monte Carlo")

st.markdown("")

colA, colB = st.columns([0.58, 0.42], gap="large")

with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Decisión & supuestos")
    st.write(rationale)
    st.markdown("**Supuestos clave**")
    st.write(
        f"- Horizonte: {n_years} años + perpetuidad\n"
        f"- g explícito: {fmt_pct(g_exp)} | g∞: {fmt_pct(g_inf)}\n"
        f"- WACC: {fmt_pct(wacc)} (Ke {fmt_pct(ke)} | Kd {fmt_pct(kd)})\n"
        f"- CAPEX₀: {fmt_pyg(capex0)}\n"
        f"- FCF Año 1 (calculado): {fmt_pyg(fcf_y1)}"
    )
    st.markdown("**Checklist Comité (automático)**")
    for label, ok in checks:
        st.write(("✅ " if ok else "❌ ") + label)
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Estructura del valor")
    st.markdown("<div class='small'>VAN = PV(FCF ciclo 1..N) + PV(TV) − CAPEX₀. TV se reporta separado.</div>", unsafe_allow_html=True)

    if np.isfinite(npv_base):
        # gráfico tipo fuente/puente (horizontal)
        df_value = {
            "Componente": ["PV FCF", "PV TV", "CAPEX₀"],
            "Monto": [pv_fcf, pv_tv, -capex0],
        }
        fig_value = px.bar(
            x=list(df_value["Monto"]),
            y=list(df_value["Componente"]),
            orientation="h",
            labels={"x": "Gs.", "y": ""},
        )
        fig_value.update_layout(
            height=220,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        fig_value.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,.07)")
        fig_value.update_yaxes(showgrid=False)

        st.plotly_chart(fig_value, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"**VAN (base): {fmt_pyg(npv_base)}**")
    else:
        st.warning("No se puede calcular TV: WACC debe ser mayor que g∞ + spread.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")
colC, colD = st.columns([0.52, 0.48], gap="large")

with colC:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Flujos proyectados (ciclo 1..N) + TV")
    st.markdown("<div class='small'>FCF por ciclo. El valor terminal (TV) se muestra separado.</div>", unsafe_allow_html=True)
    df_fcf = {"Año": [str(i) for i in years], "FCF": [float(x) for x in fcf_series]}
    fig_fcf = px.bar(df_fcf, x="Año", y="FCF")
    fig_fcf.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig_fcf.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,.07)")
    st.plotly_chart(fig_fcf, use_container_width=True, config={"displayModeBar": False})
    if np.isfinite(tv):
        st.caption(f"TV (en ciclo {n_years}): {fmt_pyg(tv)}")
    st.markdown("</div>", unsafe_allow_html=True)

with colD:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Riesgo (Monte Carlo) — distribución del VAN")
    if npv_valid.size:
        st.markdown(
            f"<div class='small'>Simulaciones: <b>{sims:,}</b> | válidas: <b>{valid_rate*100:.1f}%</b> | "
            f"P(VAN&lt;0): <b>{prob_neg*100:.1f}%</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='small'>P5/P50/P95: <b>{fmt_pyg(p5)}</b> / <b>{fmt_pyg(p50)}</b> / <b>{fmt_pyg(p95)}</b> "
            f"| Media: <b>{fmt_pyg(mean)}</b> | σ: <b>{fmt_pyg(std)}</b> | CVaR5: <b>{fmt_pyg(cvar5)}</b></div>",
            unsafe_allow_html=True,
        )

        # Histograma (plotly)
        df_hist = {"VAN": npv_valid}
        fig_hist = px.histogram(df_hist, x="VAN", nbins=40)
        fig_hist.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        fig_hist.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,.07)")
        fig_hist.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,.07)")

        # líneas P5/P50/P95
        for v in [p5, p50, p95]:
            fig_hist.add_vline(x=float(v), line_width=2, line_dash="dash", line_color="rgba(234,241,255,.65)")
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            "<div class='small'>Lectura: P5 es un escenario adverso plausible (cola izquierda), "
            "P50 es el centro, P95 es escenario favorable. CVaR5 aproxima la severidad promedio dentro del 5% peor.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Monte Carlo no generó resultados válidos (revisar WACC vs g∞, rangos y supuestos).")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='small' style='text-align:center; margin-top:8px;'>Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence. · ValuationSuite USIL (PYG)</div>", unsafe_allow_html=True)

st.divider()


# ============================================================
# Export (TXT + PDF premium)
# ============================================================
st.header("Export")

# One‑pager data for exports
onepager = rep.OnePager(
    institution="Universidad San Ignacio de Loyola (USIL)",
    program="Maestría en Administración de Negocios (MBA)",
    course="Proyectos de Inversión / Valuation",
    currency="PYG",
    project=project,
    responsible=responsible,
    report_date=date.today().isoformat(),
    verdict=verdict,
    rationale=rationale,
    # KPIs
    npv_base=float(npv_base) if np.isfinite(npv_base) else None,
    irr_base=float(irr_base) if irr_base is not None else None,
    payback_simple=float(pb_simple) if pb_simple is not None else None,
    payback_discounted=float(pb_disc) if pb_disc is not None else None,
    # assumptions
    n_years=int(n_years),
    g_exp=float(g_exp),
    g_inf=float(g_inf),
    wacc=float(wacc),
    ke=float(ke),
    kd=float(kd),
    capex0=float(capex0),
    fcf1=float(fcf_y1),
    # accounting bridge
    ebit=float(ebit),
    nopat=float(nopat),
    delta_wc=float(delta_wc),
    capex_y1=float(capex_y1),
    # deterministic valuation components
    pv_fcf=float(pv_fcf) if np.isfinite(pv_fcf) else None,
    pv_tv=float(pv_tv) if np.isfinite(pv_tv) else None,
    tv=float(tv) if np.isfinite(tv) else None,
    # Monte Carlo
    sims=int(sims),
    valid_rate=float(valid_rate) if np.isfinite(valid_rate) else None,
    prob_neg=float(prob_neg) if np.isfinite(prob_neg) else None,
    p5=float(p5) if np.isfinite(p5) else None,
    p50=float(p50) if np.isfinite(p50) else None,
    p95=float(p95) if np.isfinite(p95) else None,
    mean=float(mean) if np.isfinite(mean) else None,
    std=float(std) if np.isfinite(std) else None,
    cvar5=float(cvar5) if np.isfinite(cvar5) else None,
    checks=[(a, bool(b)) for a, b in checks],
    fcf_years=[float(x) for x in fcf_series],
)

onepager_txt = rep.build_onepager_text(onepager)

st.download_button(
    "⬇️ Descargar One‑Pager (TXT)",
    data=onepager_txt.encode("utf-8"),
    file_name="one_pager_usil.txt",
    mime="text/plain",
)

# PDF (ReportLab only)
if rep.REPORTLAB_OK:
    # histogram arrays for PDF mini‑chart
    hist_counts, hist_edges = (None, None)
    if npv_valid.size:
        hist_counts, hist_edges = np.histogram(npv_valid, bins=36)

    pdf_bytes = rep.generate_onepager_pdf(
        onepager=onepager,
        hist_counts=hist_counts,
        hist_edges=hist_edges,
    )
    st.download_button(
        "⬇️ Descargar One‑Pager (PDF premium)",
        data=pdf_bytes,
        file_name="one_pager_usil.pdf",
        mime="application/pdf",
    )
else:
    st.info("Para exportar PDF, agrega `reportlab` a requirements.txt.")
