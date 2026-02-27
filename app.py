# app.py
# ValuationApp – Evaluación Financiera (One-Pager Ejecutivo) – MBA USIL (Paraguay)
# Streamlit app (PYG, base nominal, D/E contable). Incluye DCF + Monte Carlo + Comité (defaults).

import io
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PDF (ReportLab) – import protegido
# ============================================================
REPORTLAB_OK = True
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
except Exception:
    REPORTLAB_OK = False


# ============================================================
# Config / Estilo
# ============================================================
st.set_page_config(page_title="ValuationApp – One-Pager (PYG)", layout="wide")

PREMIUM_CSS = """
<style>
:root{
  --card:#0f1b30;
  --muted:#9fb0c7;
  --text:#e9f0ff;
  --ok:#22c55e;
  --warn:#f59e0b;
  --bad:#ef4444;
  --border:rgba(255,255,255,.08);
  --shadow: 0 14px 40px rgba(0,0,0,.35);
  --radius: 18px;
}

.stApp{
  background: radial-gradient(1200px 600px at 20% 0%, rgba(91,188,255,.14), transparent 55%),
              radial-gradient(900px 500px at 95% 10%, rgba(34,197,94,.10), transparent 55%),
              linear-gradient(180deg, #080e19 0%, #0b1220 55%, #070b13 100%) !important;
}

.small-muted{ color: var(--muted); font-size: .92rem; }

.kpi-card{
  background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  border-radius: var(--radius);
  padding: 16px 16px 14px 16px;
  min-height: 92px;
}

.kpi-label{ color: var(--muted); font-size: .86rem; margin-bottom: 6px; }
.kpi-value{
  font-size: 1.55rem; font-weight: 800; line-height: 1.1;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.kpi-sub{ color: var(--muted); font-size: .82rem; margin-top: 8px; }

.panel{
  background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  border-radius: var(--radius);
  padding: 18px;
}

.badge{
  display:inline-flex; align-items:center; gap:8px;
  padding: 6px 10px; border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,.03);
  font-weight: 700;
}
.badge-ok{ color: var(--ok); }
.badge-warn{ color: var(--warn); }
.badge-bad{ color: var(--bad); }

hr{ border:none; height:1px; background: var(--border); margin: 14px 0; }
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

st.title("📊 ValuationApp — One‑Pager Ejecutivo (Evaluación Financiera, PYG)")
st.caption("Marco: DCF + evaluación probabilística (Monte Carlo). Enfoque ejecutivo (Comité Académico MBA – USIL).")

# ============================================================
# Utilidades
# ============================================================
MIN_SPREAD = 0.005  # WACC debe ser > g∞ + 0.5%
DEFAULT_SIMS = 15000

def pct(x: float) -> str:
    return f"{x:.2%}"

def fmt_pyg(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"Gs. {x:,.0f}".replace(",", ".")

def fmt_years(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:.2f} años"

def safe_float(x, default=None):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

def safe_irr(cashflows: np.ndarray):
    try:
        irr = npf.irr(cashflows)
        irr = safe_float(irr, None)
        if irr is None:
            return None
        if irr < -0.99 or irr > 5.0:
            return None
        return irr
    except Exception:
        return None

def discounted_payback(cashflows: np.ndarray, rate: float):
    if rate <= -0.99:
        return None
    cum = cashflows[0]
    if cum >= 0:
        return 0.0
    for t in range(1, len(cashflows)):
        pv = cashflows[t] / ((1 + rate) ** t)
        prev = cum
        cum += pv
        if cum >= 0:
            if pv == 0:
                return float(t)
            frac = (0 - prev) / pv
            return (t - 1) + float(frac)
    return None

def simple_payback(cashflows: np.ndarray):
    cum = cashflows[0]
    if cum >= 0:
        return 0.0
    for t in range(1, len(cashflows)):
        prev = cum
        cum += cashflows[t]
        if cum >= 0:
            cf = cashflows[t]
            if cf == 0:
                return float(t)
            frac = (0 - prev) / cf
            return (t - 1) + float(frac)
    return None

def badge(verdict: str) -> str:
    return {"APROBADO": "✅", "OBSERVADO": "⚠️", "RECHAZADO": "⛔"}.get(verdict, "—")

def kpi_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def panel_title(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f"<div class='small-muted'>{subtitle}</div>", unsafe_allow_html=True)

def committee_defaults(prob_neg: float, npv_det: float, p5: float, p50: float):
    checks = [
        ("P(VAN<0) ≤ 20%", prob_neg <= 0.20),
        ("P50(VAN) > 0", p50 > 0),
        ("P5(VAN) > 0", p5 > 0),
        ("VAN (det.) > 0", npv_det > 0),
    ]
    ok = sum(v for _, v in checks)
    if ok == len(checks):
        verdict = "APROBADO"
        rationale = (
            "El proyecto cumple criterios conservadores de creación de valor y control de downside bajo incertidumbre razonable. "
            "Se recomienda avanzar, manteniendo disciplina de supuestos y hitos de control."
        )
    elif ok == 0:
        verdict = "RECHAZADO"
        rationale = (
            "El proyecto no cumple criterios mínimos. Se recomienda reformular supuestos, estructura de inversión y mitigaciones "
            "antes de re‑presentación al Comité."
        )
    else:
        verdict = "OBSERVADO"
        rationale = (
            "El proyecto presenta señales mixtas: puede existir creación de valor, pero se identifican debilidades en indicadores o "
            "en el perfil de riesgo. Requiere reforzar evidencia, sensibilidad y mitigaciones antes de aprobación."
        )
    return verdict, rationale, checks

def explain_terms_block():
    with st.expander("📚 Explicaciones clave (no técnicas, registro MBA)"):
        st.markdown(
            """
**Flujos proyectados (Ciclo 1…N)**: FCF estimados Año 1..Año N y descontados a WACC.  
**Crecimiento explícito (g)**: crecimiento aplicado a FCF Año 2..N.  
**Crecimiento a perpetuidad (g∞)**: crecimiento de largo plazo usado solo para el Valor Terminal (TV).  
**P5/P50/P95 (Monte Carlo)**: percentiles del VAN simulado.  
**Nota**: si los flujos cambian de signo más de una vez, la TIR puede no ser única.
"""
        )

# ============================================================
# Sidebar – Inputs
# ============================================================
st.sidebar.header("🧾 Identificación")
institution = st.sidebar.text_input("Institución", "Universidad San Ignacio de Loyola (USIL)")
program = st.sidebar.text_input("Programa", "Maestría en Administración de Negocios (MBA)")
course = st.sidebar.text_input("Curso / Módulo", "Proyectos de Inversión / Valuation")
project = st.sidebar.text_input("Proyecto", "Proyecto")
responsible = st.sidebar.text_input("Responsable", "Docente: Jorge Rojas")
st.sidebar.caption("Moneda fija: **PYG (nominal)**. D/E para WACC: **valores contables**.")

st.sidebar.divider()
st.sidebar.header("0) Inversión inicial")
capex0 = st.sidebar.number_input("CAPEX Año 0 (Gs.)", value=3_500_000_000.0, step=50_000_000.0, min_value=1.0)

st.sidebar.divider()
st.sidebar.header("1) CAPM / WACC (inputs)")
rf = st.sidebar.number_input("Rf (%)", value=4.50, step=0.10) / 100
erp = st.sidebar.number_input("ERP (%)", value=5.50, step=0.10) / 100
crp = st.sidebar.number_input("Riesgo País (CRP %) en Ke", value=2.00, step=0.10) / 100
beta_u = st.sidebar.number_input("βU (desapalancada)", value=0.90, step=0.05)
tax_rate = st.sidebar.number_input("Impuesto T (%)", value=10.0, step=0.5) / 100

st.sidebar.divider()
st.sidebar.header("2) Estructura de capital (contable)")
debt = st.sidebar.number_input("Deuda D (Gs.)", value=1_400_000_000.0, step=25_000_000.0, min_value=0.0)
equity = st.sidebar.number_input("Capital propio E (Gs.)", value=2_100_000_000.0, step=25_000_000.0, min_value=1.0)
kd = st.sidebar.number_input("Costo de deuda Kd (%)", value=12.0, step=0.25) / 100

st.sidebar.divider()
st.sidebar.header("3) Horizonte explícito (FCF)")
n_years = int(st.sidebar.slider("Años de proyección (N)", min_value=3, max_value=10, value=5))
g_explicit = st.sidebar.number_input("g (%) – crecimiento explícito (Año 2…N)", value=5.0, step=0.25) / 100
g_inf = st.sidebar.number_input("g∞ (%) – crecimiento a perpetuidad (TV)", value=2.0, step=0.25) / 100

st.sidebar.divider()
st.sidebar.header("4) Año 1 – base contable simplificada")
sales1 = st.sidebar.number_input("Ventas Año 1 (Gs.)", value=8_000_000_000.0, step=50_000_000.0, min_value=0.0)
var_cost_rate = st.sidebar.number_input("Costos variables (% ventas)", value=45.0, step=1.0, min_value=0.0, max_value=100.0) / 100
fixed_cost1 = st.sidebar.number_input("Costos fijos Año 1 (Gs.)", value=2_200_000_000.0, step=25_000_000.0, min_value=0.0)
other_opex1 = st.sidebar.number_input("Otros OPEX Año 1 (Gs.)", value=300_000_000.0, step=10_000_000.0, min_value=0.0)
depr1 = st.sidebar.number_input("Depreciación Año 1 (Gs.)", value=250_000_000.0, step=10_000_000.0, min_value=0.0)
capex1 = st.sidebar.number_input("CAPEX Año 1 (Gs.)", value=120_000_000.0, step=10_000_000.0, min_value=0.0)

st.sidebar.divider()
st.sidebar.header("5) Capital de trabajo – componentes (Año 1)")
ar_days = st.sidebar.number_input("Cuentas por cobrar (días)", value=30.0, step=1.0, min_value=0.0)
inv_days = st.sidebar.number_input("Inventarios (días)", value=25.0, step=1.0, min_value=0.0)
ap_days = st.sidebar.number_input("Cuentas por pagar (días)", value=35.0, step=1.0, min_value=0.0)

st.sidebar.divider()
st.sidebar.header("6) Monte Carlo (sin semilla)")
activate_mc = st.sidebar.checkbox("Activar Monte Carlo", value=True)
sims = int(st.sidebar.slider("Simulaciones", min_value=5_000, max_value=50_000, value=DEFAULT_SIMS, step=1_000))

st.sidebar.subheader("Rangos (triangular) – incertidumbre razonable")
g_min = st.sidebar.number_input("g mínimo (%)", value=1.0, step=0.25) / 100
g_mode = st.sidebar.number_input("g base (%)", value=5.0, step=0.25) / 100
g_max = st.sidebar.number_input("g máximo (%)", value=9.0, step=0.25) / 100

auto_wacc = st.sidebar.checkbox("Auto WACC (usar WACC calculado ±2%)", value=True)

capex_min = st.sidebar.number_input("CAPEX mínimo (Gs.) – favorable", value=capex0 * 0.90, step=10_000_000.0, min_value=1.0)
capex_mode = st.sidebar.number_input("CAPEX base (Gs.) – más probable", value=capex0, step=10_000_000.0, min_value=1.0)
capex_max = st.sidebar.number_input("CAPEX máximo (Gs.) – adverso", value=capex0 * 1.15, step=10_000_000.0, min_value=1.0)

fcf_mult_min = st.sidebar.number_input("Multiplicador FCF1 mín. (adverso)", value=0.85, step=0.01, min_value=0.10)
fcf_mult_mode = st.sidebar.number_input("Multiplicador FCF1 base", value=1.00, step=0.01, min_value=0.10)
fcf_mult_max = st.sidebar.number_input("Multiplicador FCF1 máx. (favorable)", value=1.12, step=0.01, min_value=0.10)

# ============================================================
# Cálculos determinísticos
# ============================================================
beta_l = beta_u * (1.0 + (1.0 - tax_rate) * (debt / equity))
ke = rf + beta_l * erp + crp

D = max(debt, 0.0)
E = max(equity, 1.0)
w_d = D / (D + E)
w_e = E / (D + E)
wacc = w_e * ke + w_d * kd * (1.0 - tax_rate)

var_cost1 = sales1 * var_cost_rate
cogs1 = var_cost1
ebit1 = sales1 - var_cost1 - fixed_cost1 - other_opex1 - depr1
nopat1 = ebit1 * (1.0 - tax_rate)

ar1 = sales1 / 365.0 * ar_days
inv1 = cogs1 / 365.0 * inv_days
ap1 = cogs1 / 365.0 * ap_days
nwc1 = ar1 + inv1 - ap1

sales0 = sales1 / max(1e-9, (1.0 + g_explicit))
cogs0 = sales0 * var_cost_rate
ar0 = sales0 / 365.0 * ar_days
inv0 = cogs0 / 365.0 * inv_days
ap0 = cogs0 / 365.0 * ap_days
nwc0 = ar0 + inv0 - ap0
delta_nwc1 = nwc1 - nwc0

fcf1 = nopat1 + depr1 - capex1 - delta_nwc1

fcf_years = np.array([fcf1 * ((1.0 + g_explicit) ** (t - 1)) for t in range(1, n_years + 1)], dtype=float)
years = np.arange(1, n_years + 1)

tv_ok = (wacc > g_inf + MIN_SPREAD)
tv = (fcf_years[-1] * (1.0 + g_inf)) / max(1e-12, (wacc - g_inf))

pv_fcf = np.sum(fcf_years / ((1.0 + wacc) ** years))
pv_tv = tv / ((1.0 + wacc) ** n_years)
npv_det = pv_fcf + pv_tv - capex0

cashflows_det = np.concatenate(([-capex0], fcf_years[:-1], [fcf_years[-1] + tv]))
irr_det = safe_irr(cashflows_det)
pb_simple = simple_payback(np.concatenate(([-capex0], fcf_years)))
pb_disc = discounted_payback(np.concatenate(([-capex0], fcf_years)), wacc)

# ============================================================
# Monte Carlo
# ============================================================
@st.cache_data(show_spinner=False)
def run_monte_carlo(
    sims: int,
    fcf1: float,
    n_years: int,
    g_inf: float,
    min_spread: float,
    g_min: float, g_mode: float, g_max: float,
    w_min: float, w_mode: float, w_max: float,
    capex_min: float, capex_mode: float, capex_max: float,
    mult_min: float, mult_mode: float, mult_max: float,
):
    rng = np.random.default_rng()
    g_s = rng.triangular(g_min, g_mode, g_max, sims)
    w_s = rng.triangular(w_min, w_mode, w_max, sims)
    capex_s = rng.triangular(capex_min, capex_mode, capex_max, sims)
    mult_s = rng.triangular(mult_min, mult_mode, mult_max, sims)

    yrs = np.arange(1, n_years + 1)
    fcf1_s = fcf1 * mult_s
    fcf_paths = fcf1_s[:, None] * (1.0 + g_s)[:, None] ** (yrs[None, :] - 1)

    valid = w_s > (g_inf + min_spread)
    npv_s = np.full(sims, np.nan)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return npv_s, idx

    fcf_valid = fcf_paths[idx, :]
    w_valid = w_s[idx]
    capex_valid = capex_s[idx]

    tv_valid = (fcf_valid[:, -1] * (1.0 + g_inf)) / (w_valid - g_inf)
    fcf_valid[:, -1] = fcf_valid[:, -1] + tv_valid

    discount = (1.0 + w_valid)[:, None] ** (yrs[None, :])
    pv = np.sum(fcf_valid / discount, axis=1)

    npv_s[idx] = pv - capex_valid
    return npv_s, idx

prob_neg = None
p5 = p50 = p95 = None
npv_valid = None

if activate_mc:
    if auto_wacc:
        w_min = max(0.0001, wacc - 0.02)
        w_mode = max(0.0001, wacc)
        w_max = max(0.0001, wacc + 0.02)
    else:
        w_min, w_mode, w_max = 0.09, 0.11, 0.13

    npv_s, idx = run_monte_carlo(
        sims=sims,
        fcf1=fcf1,
        n_years=n_years,
        g_inf=g_inf,
        min_spread=MIN_SPREAD,
        g_min=g_min, g_mode=g_mode, g_max=g_max,
        w_min=w_min, w_mode=w_mode, w_max=w_max,
        capex_min=capex_min, capex_mode=capex_mode, capex_max=capex_max,
        mult_min=fcf_mult_min, mult_mode=fcf_mult_mode, mult_max=fcf_mult_max,
    )
    npv_valid = npv_s[np.isfinite(npv_s)]
    if npv_valid.size >= 200:
        prob_neg = float(np.mean(npv_valid < 0))
        p5, p50, p95 = np.percentile(npv_valid, [5, 50, 95])

# Comité
if prob_neg is not None and p50 is not None and p5 is not None:
    verdict, rationale, checks = committee_defaults(prob_neg, npv_det, p5, p50)
else:
    verdict, rationale, checks = ("—", "Active Monte Carlo para obtener un dictamen probabilístico.", [])

# ============================================================
# Layout – Dashboard
# ============================================================
left, right = st.columns([1.25, 1.0], gap="large")

with left:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    panel_title("Decisión & supuestos", "Snapshot ejecutivo para toma de decisiones (PYG).")

    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
    with c1:
        kpi_card("VAN (det.)", fmt_pyg(npv_det), f"WACC {pct(wacc)}")
    with c2:
        kpi_card("TIR (det.)", pct(irr_det) if irr_det is not None else "N/A", "Flujos no conv.")
    with c3:
        kpi_card("Payback", fmt_years(pb_simple), "No desc.")
    with c4:
        kpi_card("Payback (desc.)", fmt_years(pb_disc), "Con WACC")
    with c5:
        kpi_card("P(VAN<0)", f"{prob_neg:.1%}" if prob_neg is not None else "—", "Monte Carlo")
    with c6:
        kpi_card("P5 (VAN)", fmt_pyg(p5) if p5 is not None else "—", "Downside")

    st.markdown("<hr/>", unsafe_allow_html=True)

    if verdict != "—":
        cls = "badge-ok" if verdict == "APROBADO" else ("badge-warn" if verdict == "OBSERVADO" else "badge-bad")
        st.markdown(f"<div class='badge {cls}'>{badge(verdict)} {verdict}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted' style='margin-top:8px'>{rationale}</div>", unsafe_allow_html=True)

        st.markdown("**Criterios aplicados (defaults):**")
        for name, ok in checks:
            st.write(f"- {name} → {'Cumple ✅' if ok else 'No cumple ❌'}")
    else:
        st.info(rationale)

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("**Supuestos clave**")
    k1, k2, k3, k4 = st.columns(4, gap="small")
    with k1:
        st.write(f"**Horizonte:** {n_years} años")
        st.write(f"**g (explícito):** {pct(g_explicit)}")
        st.write(f"**g∞ (perpet.):** {pct(g_inf)}")
    with k2:
        st.write(f"**WACC:** {pct(wacc)}")
        st.write(f"**Ke:** {pct(ke)}")
        st.write(f"**Kd:** {pct(kd)}")
    with k3:
        st.write(f"**CAPEX 0:** {fmt_pyg(capex0)}")
        st.write(f"**FCF Año 1:** {fmt_pyg(fcf1)}")
        st.write(f"**TV (Año {n_years}):** {fmt_pyg(tv)}")
    with k4:
        st.write(f"**βU:** {beta_u:.2f}")
        st.write(f"**βL:** {beta_l:.2f}")
        st.write(f"**CRP:** {pct(crp)}")

    if not tv_ok:
        st.error("Inconsistencia: WACC debe ser mayor que g∞ + 0.5%. Ajuste WACC o g∞.")

    explain_terms_block()
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    panel_title("Estructura del valor", "Descomposición del VAN en componentes (PV FCF, PV TV, CAPEX).")

    wf = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative", "relative", "total"],
        y=["PV FCF (1..N)", "PV TV", "VAN (det.)"],
        x=[pv_fcf, pv_tv, pv_fcf + pv_tv - capex0],
        text=[fmt_pyg(pv_fcf), fmt_pyg(pv_tv), fmt_pyg(npv_det)],
        textposition="outside",
        connector={"line": {"width": 1}},
    ))
    wf.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Gs.", yaxis_title="", showlegend=False)
    st.plotly_chart(wf, use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    panel_title("Flujos proyectados (Ciclo 1…N) + TV", "TV se reporta por separado (evita confusión del VAN).")

    df_flows = pd.DataFrame({
        "Año": years,
        "FCF (Gs.)": fcf_years,
        "PV FCF (Gs.)": fcf_years / ((1.0 + wacc) ** years),
    })
    df_show = df_flows.copy()
    df_show["FCF (Gs.)"] = df_show["FCF (Gs.)"].map(fmt_pyg)
    df_show["PV FCF (Gs.)"] = df_show["PV FCF (Gs.)"].map(fmt_pyg)
    st.dataframe(df_show, use_container_width=True, hide_index=True)
    st.write(f"**Valor Terminal (TV) en año {n_years}:** {fmt_pyg(tv)}")
    st.write(f"**PV del TV:** {fmt_pyg(pv_tv)}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='panel'>", unsafe_allow_html=True)
panel_title("Riesgo (Monte Carlo) — lectura ejecutiva", "Distribución del VAN bajo incertidumbre razonable.")

if activate_mc and npv_valid is not None and npv_valid.size >= 200:
    hist = px.histogram(pd.DataFrame({"VAN": npv_valid}), x="VAN", nbins=40, opacity=0.95)
    hist.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="VAN (Gs.)", yaxis_title="Frecuencia")
    for val, name in [(p5, "P5"), (p50, "P50"), (p95, "P95")]:
        hist.add_vline(x=val, line_width=2, line_dash="dash", annotation_text=f"{name}: {fmt_pyg(val)}", annotation_position="top right")
    st.plotly_chart(hist, use_container_width=True)

    colA, colB, colC, colD = st.columns(4, gap="small")
    with colA:
        kpi_card("P50 (VAN)", fmt_pyg(p50), "Mediana")
    with colB:
        kpi_card("P5 (VAN)", fmt_pyg(p5), "Adverso")
    with colC:
        kpi_card("P95 (VAN)", fmt_pyg(p95), "Favorable")
    with colD:
        kpi_card("P(VAN<0)", f"{prob_neg:.1%}", "Downside")
else:
    st.info("Monte Carlo desactivado o con pocas simulaciones válidas. Active Monte Carlo y verifique WACC > g∞.")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Reporte One‑Pager (texto + PDF)
# ============================================================
@dataclass
class OnePager:
    institution: str
    program: str
    course: str
    project: str
    responsible: str
    wacc: float
    ke: float
    kd: float
    beta_u: float
    beta_l: float
    capex0: float
    fcf1: float
    g: float
    g_inf: float
    n_years: int
    pv_fcf: float
    pv_tv: float
    tv: float
    npv_det: float
    irr_det: float | None
    pb_simple: float | None
    pb_disc: float | None
    sims: int | None
    prob_neg: float | None
    p5: float | None
    p50: float | None
    p95: float | None
    verdict: str
    rationale: str

def build_onepager_text(r: OnePager, df_flows: pd.DataFrame) -> str:
    irr_txt = pct(r.irr_det) if r.irr_det is not None else "N/A"
    lines = []
    lines.append("ONE‑PAGER EJECUTIVO — EVALUACIÓN FINANCIERA")
    lines.append(f"{r.institution} — {r.program} — {r.course}")
    lines.append(f"Fecha: {date.today().isoformat()}")
    lines.append("")
    lines.append(f"Proyecto: {r.project}")
    lines.append(f"Responsable: {r.responsible}")
    lines.append("")
    lines.append(f"DICTAMEN: {r.verdict} {badge(r.verdict)}")
    lines.append(r.rationale)
    lines.append("")
    lines.append("KPIs (determinístico)")
    lines.append(f"- VAN: {fmt_pyg(r.npv_det)}")
    lines.append(f"- TIR: {irr_txt}")
    lines.append(f"- Payback: {fmt_years(r.pb_simple)} | Payback descontado: {fmt_years(r.pb_disc)}")
    lines.append("")
    lines.append("Flujos proyectados (FCF) — ciclo 1..N")
    for _, row in df_flows.iterrows():
        lines.append(f"- Año {int(row['Año'])}: {fmt_pyg(row['FCF (Gs.)'])}")
    lines.append("")
    lines.append("Estructura del valor")
    lines.append(f"- PV(FCF 1..N): {fmt_pyg(r.pv_fcf)}")
    lines.append(f"- TV año {r.n_years}: {fmt_pyg(r.tv)} | PV(TV): {fmt_pyg(r.pv_tv)}")
    lines.append("")
    lines.append("Riesgo (Monte Carlo)")
    if r.prob_neg is None:
        lines.append("- Monte Carlo no disponible.")
    else:
        lines.append(f"- Simulaciones: {r.sims:,}")
        lines.append(f"- P(VAN<0): {r.prob_neg:.1%}")
        lines.append(f"- P5: {fmt_pyg(r.p5)} | P50: {fmt_pyg(r.p50)} | P95: {fmt_pyg(r.p95)}")
    lines.append("")
    lines.append("Nota académica: resultados dependen de supuestos y evidencia; no sustituyen due diligence.")
    return "\n".join(lines)

def generate_onepager_pdf(report: OnePager, df_flows, npv_samples=None) -> bytes:
    """
    One-pager ejecutivo estilo dashboard (oscuro), en 1 página.
    - KPIs en "cards"
    - Caja de decisión
    - Puente de valor (PV FCF + PV TV – CAPEX)
    - Gráfico de flujos (FCF) + TV separado
    - Histograma del VAN (Monte Carlo) con P5/P50/P95 (si hay muestras)
    """
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab no está disponible. Agregue `reportlab` a requirements.txt.")

    # Imports locales para evitar dependencias si no se usa PDF
    from reportlab.lib import colors
    from reportlab.lib.utils import ImageReader
    import matplotlib.pyplot as plt

    def fig_to_png_bytes(fig) -> bytes:
        bio = io.BytesIO()
        fig.savefig(bio, format="png", dpi=170, bbox_inches="tight", transparent=True)
        plt.close(fig)
        bio.seek(0)
        return bio.getvalue()

    # ------------------------------------------------------------
    # Gráfico 1: Flujos (FCF ciclo 1..N) + TV separado
    # ------------------------------------------------------------
    years = list(df_flows["Año"].values)
    fcf_vals = list(df_flows["FCF"].values)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.bar(years, fcf_vals)
    ax1.set_title("Flujos proyectados (FCF) – ciclo 1..N", fontsize=10)
    ax1.set_xlabel("Año")
    ax1.set_ylabel("FCF (Gs.)")
    ax1.tick_params(axis='both', labelsize=8)
    flows_png = fig_to_png_bytes(fig1)

    # ------------------------------------------------------------
    # Gráfico 2: Histograma VAN (Monte Carlo) – opcional
    # ------------------------------------------------------------
    hist_png = None
    if npv_samples is not None:
        v = np.asarray(npv_samples, dtype=float)
        v = v[np.isfinite(v)]
        if v.size >= 200:
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2.hist(v, bins=30)
            ax2.set_title("Distribución del VAN (Monte Carlo)", fontsize=10)
            ax2.set_xlabel("VAN (Gs.)")
            ax2.set_ylabel("Frecuencia")
            ax2.tick_params(axis='both', labelsize=8)
            if report.p5 is not None: ax2.axvline(report.p5, linestyle="--")
            if report.p50 is not None: ax2.axvline(report.p50, linestyle="--")
            if report.p95 is not None: ax2.axvline(report.p95, linestyle="--")
            hist_png = fig_to_png_bytes(fig2)

    # ------------------------------------------------------------
    # Lienzo PDF
    # ------------------------------------------------------------
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter

    # Paleta (dashboard oscuro)
    bg = colors.HexColor("#0B1220")
    card = colors.HexColor("#121B2D")
    card2 = colors.HexColor("#0F172A")
    stroke = colors.HexColor("#22304A")
    text = colors.HexColor("#E6EEF8")
    muted = colors.HexColor("#A7B3C7")
    green = colors.HexColor("#22C55E")
    amber = colors.HexColor("#F59E0B")
    red = colors.HexColor("#EF4444")
    blue = colors.HexColor("#60A5FA")

    # Fondo
    c.setFillColor(bg)
    c.rect(0, 0, W, H, stroke=0, fill=1)

    pad = 28
    x0, y0 = pad, pad
    usable_w, usable_h = W - 2 * pad, H - 2 * pad

    # Header
    c.setFillColor(text)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x0, H - pad - 8, "ONE‑PAGER EJECUTIVO — EVALUACIÓN FINANCIERA")
    c.setFont("Helvetica", 9.5)
    c.setFillColor(muted)
    c.drawString(x0, H - pad - 24, f"{report.institution} — {report.program} — {report.course}")
    c.drawRightString(W - pad, H - pad - 24, f"Fecha: {date.today().isoformat()}")

    # Helper: card
    def draw_card(x, y, w, h, title, value, subtitle=None, accent=None):
        c.setFillColor(card)
        c.setStrokeColor(stroke)
        c.roundRect(x, y, w, h, 10, stroke=1, fill=1)
        c.setFillColor(muted)
        c.setFont("Helvetica", 8.5)
        c.drawString(x + 12, y + h - 18, title)
        c.setFillColor(text)
        c.setFont("Helvetica-Bold", 13.5)
        c.drawString(x + 12, y + h - 38, value)
        if subtitle:
            c.setFillColor(muted)
            c.setFont("Helvetica", 8.5)
            c.drawString(x + 12, y + 12, subtitle)
        if accent:
            c.setStrokeColor(accent)
            c.setLineWidth(2.2)
            c.line(x + 2, y + 2, x + w - 2, y + 2)
            c.setLineWidth(1)

    # Dictamen color
    verdict_color = green if report.verdict == "APROBADO" else amber if report.verdict == "OBSERVADO" else red

    # KPI row
    kpi_y = H - pad - 88
    kpi_h = 58
    gap = 10
    kpi_w = (usable_w - gap * 5) / 6

    draw_card(x0 + 0*(kpi_w+gap), kpi_y, kpi_w, kpi_h, "VAN (base)", fmt_pyg(report.npv_det), "Determinístico", blue)
    draw_card(x0 + 1*(kpi_w+gap), kpi_y, kpi_w, kpi_h, "TIR (base)", pct(report.irr_det) if report.irr_det is not None else "N/A", "Determinístico", blue)
    draw_card(x0 + 2*(kpi_w+gap), kpi_y, kpi_w, kpi_h, "Payback", f"{report.pb_simple:.2f} años" if report.pb_simple is not None else "N/A", "Simple", blue)
    draw_card(x0 + 3*(kpi_w+gap), kpi_y, kpi_w, kpi_h, "Payback (desc.)", f"{report.pb_disc:.2f} años" if report.pb_disc is not None else "N/A", "Descontado", blue)

    prob_text = f"{report.prob_neg:.1%}" if report.prob_neg is not None else "—"
    p50_text = fmt_pyg(report.p50) if report.p50 is not None else "—"
    draw_card(x0 + 4*(kpi_w+gap), kpi_y, kpi_w, kpi_h, "P(VAN<0)", prob_text, "Downside", verdict_color)
    draw_card(x0 + 5*(kpi_w+gap), kpi_y, kpi_w, kpi_h, "P50 (VAN)", p50_text, "Monte Carlo", verdict_color)

    # Mid row: Decisión & estructura del valor
    mid_y = kpi_y - 148
    mid_h = 130
    left_w = (usable_w - gap) * 0.52
    right_w = usable_w - gap - left_w

    # Left: decisión
    c.setFillColor(card2)
    c.setStrokeColor(stroke)
    c.roundRect(x0, mid_y, left_w, mid_h, 12, stroke=1, fill=1)
    c.setFillColor(text)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x0 + 12, mid_y + mid_h - 22, "Decisión & supuestos")

    c.setFont("Helvetica", 9.2)
    c.setFillColor(muted)
    lines = [
        f"Proyecto: {report.project} | Responsable: {report.responsible}",
        f"CAPEX Año 0: {fmt_pyg(report.capex0)} | FCF Año 1: {fmt_pyg(report.fcf1)}",
        f"WACC: {pct(report.wacc)} | g (expl.): {pct(report.g)} | g∞: {pct(report.g_inf)} | Horizonte: {report.n_years} años",
    ]
    if report.sims is not None:
        p5t = fmt_pyg(report.p5) if report.p5 is not None else "—"
        p95t = fmt_pyg(report.p95) if report.p95 is not None else "—"
        lines.append(f"Simulaciones: {report.sims:,} | P5: {p5t} | P95: {p95t}")
    yy = mid_y + mid_h - 42
    for ln in lines:
        c.drawString(x0 + 12, yy, ln)
        yy -= 14

    # Dictamen pill
    c.setFillColor(verdict_color)
    c.roundRect(x0 + 12, mid_y + 18, 120, 22, 10, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawCentredString(x0 + 72, mid_y + 25, f"DICTAMEN: {report.verdict}")

    # rationale (corto, en una línea)
    c.setFillColor(muted)
    c.setFont("Helvetica", 9.0)
    rationale_line = report.rationale.strip()
    if len(rationale_line) > 92:
        rationale_line = rationale_line[:92].rstrip() + "…"
    c.drawString(x0 + 140, mid_y + 24, rationale_line)

    # Right: estructura del valor
    xr = x0 + left_w + gap
    c.setFillColor(card2)
    c.setStrokeColor(stroke)
    c.roundRect(xr, mid_y, right_w, mid_h, 12, stroke=1, fill=1)

    c.setFillColor(text)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(xr + 12, mid_y + mid_h - 22, "Estructura del valor")
    c.setFillColor(muted)
    c.setFont("Helvetica", 9.0)
    c.drawString(xr + 12, mid_y + mid_h - 42, "VAN = PV(FCF ciclo 1..N) + PV(TV) – CAPEX")

    pv_fcf = max(report.pv_fcf, 0.0)
    pv_tv = max(report.pv_tv, 0.0)
    cap = max(report.capex0, 0.0)
    total = pv_fcf + pv_tv + cap
    total = total if total > 0 else 1.0

    bar_x = xr + 12
    bar_y = mid_y + 42
    bar_w = right_w - 24
    bar_h = 16

    c.setFillColor(stroke)
    c.roundRect(bar_x, bar_y, bar_w, bar_h, 8, stroke=0, fill=1)

    seg_fcf = bar_w * (pv_fcf / total)
    seg_tv = bar_w * (pv_tv / total)

    c.setFillColor(blue)
    c.roundRect(bar_x, bar_y, seg_fcf, bar_h, 8, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#93C5FD"))
    c.rect(bar_x + seg_fcf, bar_y, seg_tv, bar_h, stroke=0, fill=1)

    c.setFillColor(muted)
    c.setFont("Helvetica", 8.5)
    c.drawString(bar_x, bar_y - 12, f"PV FCF: {fmt_pyg(pv_fcf)}")
    c.drawRightString(bar_x + bar_w, bar_y - 12, f"PV TV: {fmt_pyg(pv_tv)}")

    c.setFillColor(text)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(xr + 12, mid_y + 18, f"VAN: {fmt_pyg(report.npv_det)}")

    # Bottom row: charts
    bot_y = y0 + 22
    bot_h = mid_y - bot_y - 18
    bot_left_w = (usable_w - gap) * 0.52
    bot_right_w = usable_w - gap - bot_left_w

    # Flows chart card
    c.setFillColor(card)
    c.setStrokeColor(stroke)
    c.roundRect(x0, bot_y, bot_left_w, bot_h, 12, stroke=1, fill=1)
    c.setFillColor(text)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x0 + 12, bot_y + bot_h - 22, "Flujos proyectados (ciclo 1..N) + TV")
    c.setFillColor(muted)
    c.setFont("Helvetica", 8.5)
    c.drawString(x0 + 12, bot_y + bot_h - 38, "FCF por ciclo (TV se reporta separado)")

    img1 = ImageReader(io.BytesIO(flows_png))
    c.drawImage(img1, x0 + 12, bot_y + 18, width=bot_left_w - 24, height=bot_h - 62, mask="auto")
    c.setFillColor(muted)
    c.setFont("Helvetica", 8.5)
    c.drawString(x0 + 12, bot_y + 8, f"TV (en ciclo {report.n_years}): {fmt_pyg(report.tv)}")

    # Risk chart card
    xr2 = x0 + bot_left_w + gap
    c.setFillColor(card)
    c.setStrokeColor(stroke)
    c.roundRect(xr2, bot_y, bot_right_w, bot_h, 12, stroke=1, fill=1)
    c.setFillColor(text)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(xr2 + 12, bot_y + bot_h - 22, "Riesgo (Monte Carlo) – lectura ejecutiva")

    c.setFillColor(muted)
    c.setFont("Helvetica", 9.0)
    if report.p50 is not None:
        c.drawString(xr2 + 12, bot_y + bot_h - 42, f"P50(VAN): {fmt_pyg(report.p50)}")
    if report.p5 is not None or report.p95 is not None:
        p5t = fmt_pyg(report.p5) if report.p5 is not None else "—"
        p95t = fmt_pyg(report.p95) if report.p95 is not None else "—"
        c.drawString(xr2 + 12, bot_y + bot_h - 56, f"P5: {p5t}  |  P95: {p95t}")
    if report.prob_neg is not None and report.sims is not None:
        c.drawString(xr2 + 12, bot_y + bot_h - 70, f"P(VAN<0): {report.prob_neg:.1%}  |  Simulaciones: {report.sims:,}")

    if hist_png is not None:
        img2 = ImageReader(io.BytesIO(hist_png))
        c.drawImage(img2, xr2 + 12, bot_y + 18, width=bot_right_w - 24, height=bot_h - 98, mask="auto")
    else:
        c.setFillColor(muted)
        c.setFont("Helvetica-Oblique", 9.0)
        c.drawString(xr2 + 12, bot_y + 40, "Histograma no disponible (sin muestras suficientes).")

    # Footer
    c.setFillColor(muted)
    c.setFont("Helvetica", 7.8)
    c.drawString(x0, y0 - 6,
                 "Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence.")
    c.drawRightString(W - pad, y0 - 6, "ValuationApp — One‑Pager Dashboard (PYG)")

    c.showPage()
    c.save()
    return buf.getvalue()


report = OnePager(
    institution=institution, program=program, course=course,
    project=project, responsible=responsible,
    wacc=wacc, ke=ke, kd=kd, beta_u=beta_u, beta_l=beta_l,
    capex0=capex0, fcf1=fcf1, g=g_explicit, g_inf=g_inf, n_years=n_years,
    pv_fcf=pv_fcf, pv_tv=pv_tv, tv=tv,
    npv_det=npv_det, irr_det=irr_det, pb_simple=pb_simple, pb_disc=pb_disc,
    sims=sims if activate_mc else None,
    prob_neg=prob_neg, p5=p5, p50=p50, p95=p95,
    verdict=verdict if verdict != "—" else "SIN DICTAMEN",
    rationale=rationale if verdict != "—" else rationale,
)

st.markdown("<div class='panel'>", unsafe_allow_html=True)
panel_title("Reporte (One‑Pager)", "Descargable (TXT/PDF).")
txt = build_onepager_text(report, pd.DataFrame({"Año": years, "FCF (Gs.)": fcf_years}))
st.text_area("One‑Pager (texto)", txt, height=240)

st.download_button("⬇️ Descargar One‑Pager (TXT)", data=txt.encode("utf-8"), file_name="one_pager_valuation_pyg.txt", mime="text/plain")

if REPORTLAB_OK:
    try:
        pdf_bytes = generate_onepager_pdf(report, pd.DataFrame({"Año": years, "FCF": fcf_years}), npv_samples=npv_valid)
        st.download_button("⬇️ Descargar One‑Pager (PDF)", data=pdf_bytes, file_name="one_pager_valuation_pyg.pdf", mime="application/pdf")
    except Exception as e:
        st.warning(f"No se pudo generar PDF: {e}")
else:
    st.info("PDF no disponible: agregue `reportlab` a requirements.txt para exportar.")

st.markdown("</div>", unsafe_allow_html=True)

st.caption("Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence.")
