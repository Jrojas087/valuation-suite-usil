import streamlit as st
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# ValuationApp - MBA Edition (v4 BULLETPROOF + COMITÉ + ACCIONES)
# Enfoque: DCF (Enterprise Value) con WACC + CAPM (+ CRP opcional)
# Incluye:
# - CAPEX Año 0 (NPV/IRR correctos)
# - Guardrails (WACC > g∞ + spread)
# - Sensibilidad NPV vs WACC & g∞
# - Monte Carlo (triangular) NPV con incertidumbre en g, WACC y CAPEX
#   + Auto-WACC (±2% del WACC calculado) + override
#   + caching (mejor performance en Streamlit Cloud)
# - Modo Comité (veredicto automático)
# - Tarjeta "Acción recomendada" (orienta discusión ejecutiva)
# - Proxy tipo Altman (pedagógico)
# ============================================================

st.set_page_config(page_title="ValuationApp - MBA Edition", layout="wide")
st.title("📊 ValuationApp: Project Finance & Valuation (DCF / Enterprise)")
st.markdown("### Universidad San Ignacio de Loyola - MBA Edition")

# ------------------------
# CONSTANTES / GUARDRAILS
# ------------------------
MIN_SPREAD = 0.005  # 0.5%: guardrail de estabilidad para TV y Monte Carlo

# ------------------------
# SIDEBAR: INPUTS (BASE)
# ------------------------
st.sidebar.header("0) Inversión inicial del proyecto")
capex0 = st.sidebar.number_input("CAPEX / Inversión inicial (Año 0) $", value=500000, step=10000)

st.sidebar.divider()
st.sidebar.header("1) Costo de Equity (CAPM)")

rf = st.sidebar.number_input("Tasa libre de riesgo (Rf %) ", value=4.5, step=0.1) / 100
erp = st.sidebar.number_input("Equity Risk Premium (ERP %) ", value=5.5, step=0.1) / 100

use_crp = st.sidebar.checkbox("Incluir Riesgo País (CRP) en Ke", value=True)
crp = st.sidebar.number_input("Riesgo País (CRP %) ", value=2.0, step=0.1) / 100 if use_crp else 0.0

beta_u = st.sidebar.number_input("Beta desapalancada (βU) ", value=0.90, step=0.05)
tax_rate = st.sidebar.number_input("Tasa impuestos (T %) ", value=10.0, step=0.5) / 100

st.sidebar.divider()
st.sidebar.header("2) Estructura de capital (para WACC)")

deuda = st.sidebar.number_input("Deuda total (D) $", value=400000, step=10000)
equity = st.sidebar.number_input("Patrimonio (E) $", value=600000, step=10000)
cost_debt = st.sidebar.number_input("Costo deuda (Kd %) ", value=7.0, step=0.1) / 100

st.sidebar.divider()
st.sidebar.header("3) Proyecciones de flujos (FCF)")

n_years = st.sidebar.slider("Años de proyección", 1, 15, 5)
fcf_y1 = st.sidebar.number_input("FCF Año 1 ($)", value=100000, step=5000)
growth_rate = st.sidebar.number_input("Crecimiento anual FCF (g %) ", value=5.0, step=0.1) / 100
terminal_growth = st.sidebar.number_input("Crecimiento perpetuo (g∞ %) ", value=2.0, step=0.1) / 100

# ------------------------
# VALIDACIONES TEMPRANAS (BASE)
# ------------------------
if (deuda + equity) <= 0:
    st.error("La suma Deuda + Patrimonio debe ser mayor que 0.")
    st.stop()

if equity <= 0:
    st.error("El Patrimonio (E) debe ser mayor que 0 para calcular Beta apalancada.")
    st.stop()

if capex0 < 0:
    st.error("CAPEX debe ser un valor positivo (la app asume salida de caja en Año 0).")
    st.stop()

# ------------------------
# CÁLCULO BASE: CAPM → WACC → DCF
# ------------------------
total_val = deuda + equity

# 1) Beta apalancada (Hamada): βL = βU * [1 + (1 - T) * (D/E)]
beta_l = beta_u * (1 + (1 - tax_rate) * (deuda / equity))

# 2) Ke (CAPM + CRP opcional)
ke = rf + beta_l * erp + crp

# 3) WACC
wacc = ((equity / total_val) * ke) + ((deuda / total_val) * cost_debt * (1 - tax_rate))

# Guardrail para TV
if wacc <= terminal_growth + MIN_SPREAD:
    st.error(
        "Guardrail: WACC debe ser mayor que el crecimiento perpetuo (g∞) por al menos 0.5%. "
        "Ajustá WACC, g∞, estructura de capital o parámetros CAPM."
    )
    st.stop()

# 4) Proyección FCF determinística (Año 1..n)
years = list(range(1, n_years + 1))
cash_flows = [fcf_y1 * (1 + growth_rate) ** (i - 1) for i in years]

# 5) Valor Terminal (Gordon)
tv = (cash_flows[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)

# 6) Flujos del proyecto y NPV/IRR
flows = cash_flows.copy()
flows[-1] += tv

npv = npf.npv(wacc, [-capex0] + flows)
irr = npf.irr([-capex0] + flows)

def format_irr(x: float):
    """Saneado simple: IRR puede ser múltiple o absurda en casos borde."""
    if x is None or np.isnan(x) or np.isinf(x):
        return "N/A"
    if x < -1.0 or x > 2.0:  # <-100% o >200% (regla práctica)
        return "N/A"
    return f"{x:.2%}"

# ------------------------
# KPIs BASE
# ------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("βL (Beta apalancada)", f"{beta_l:.2f}")
k2.metric("Ke (CAPM)", f"{ke:.2%}")
k3.metric("WACC", f"{wacc:.2%}")
k4.metric("TV", f"${tv:,.0f}")
k5.metric("VAN (NPV)", f"${npv:,.0f}")

st.divider()

# ------------------------
# Gráfico: FCF + TV
# ------------------------
fig_flows = go.Figure()
fig_flows.add_trace(go.Bar(x=years, y=cash_flows, name="FCF (Años 1..n)"))
fig_flows.add_trace(go.Bar(x=[n_years], y=[tv], name="Valor Terminal (TV)"))
fig_flows.update_layout(
    title="Proyección de Flujos de Caja (FCF) + Valor Terminal",
    barmode="stack",
    xaxis_title="Año",
    yaxis_title="USD"
)
st.plotly_chart(fig_flows, use_container_width=True)

# ------------------------
# Indicadores base
# ------------------------
st.subheader("📌 Indicadores del Proyecto (Base determinística)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("CAPEX (Año 0)", f"${capex0:,.0f}")
c2.metric("TIR (IRR)", format_irr(irr))
c3.metric("VAN (NPV)", f"${npv:,.0f}")
c4.metric("Spread WACC - g∞", f"{(wacc - terminal_growth):.2%}")

# ------------------------
# Sensibilidad NPV vs WACC & g∞
# ------------------------
st.divider()
st.subheader("🧪 Sensibilidad: VAN vs WACC & Crecimiento Perpetuo (g∞)")

wacc_range = np.linspace(max(0.001, wacc - 0.02), wacc + 0.02, 5)
g_range = np.linspace(max(-0.02, terminal_growth - 0.01), terminal_growth + 0.01, 5)

sens_matrix = []
for g in g_range:
    row = []
    for w in wacc_range:
        if w <= g + MIN_SPREAD:
            row.append(np.nan)
            continue
        temp_tv = (cash_flows[-1] * (1 + g)) / (w - g)
        temp_flows = cash_flows.copy()
        temp_flows[-1] += temp_tv
        row.append(npf.npv(w, [-capex0] + temp_flows))
    sens_matrix.append(row)

fig_heat = px.imshow(
    sens_matrix,
    x=[f"{x:.2%}" for x in wacc_range],
    y=[f"{y:.2%}" for y in g_range],
    labels=dict(x="WACC", y="Crecimiento perpetuo (g∞)", color="VAN"),
    aspect="auto"
)
st.plotly_chart(fig_heat, use_container_width=True)

# ============================================================
# MONTE CARLO + MODO COMITÉ + ACCIONES
# ============================================================
st.sidebar.divider()
st.sidebar.header("4) Monte Carlo + Modo Comité")

enable_mc = st.sidebar.checkbox("Activar Monte Carlo", value=False)
mc_sims = st.sidebar.slider("Nº de simulaciones", 1000, 50000, 10000, step=1000)

st.sidebar.subheader("Variable 1: g (Triangular)")
g_min = st.sidebar.number_input("g mínimo (%)", value=max(-5.0, (growth_rate * 100) - 4.0), step=0.1) / 100
g_mode = st.sidebar.number_input("g base (%)", value=(growth_rate * 100), step=0.1) / 100
g_max = st.sidebar.number_input("g máximo (%)", value=(growth_rate * 100) + 4.0, step=0.1) / 100

st.sidebar.subheader("Variable 2: WACC (Triangular)")
auto_wacc = st.sidebar.checkbox("Auto WACC (±2% del WACC calculado)", value=True)
if auto_wacc:
    w_min = max(0.001, wacc - 0.02)
    w_mode = wacc
    w_max = wacc + 0.02
    st.sidebar.caption(f"WACC base actual: {wacc:.2%} (se usa como modo)")
else:
    w_min = st.sidebar.number_input("WACC mínimo (%)", value=max(0.1, (wacc * 100) - 2.0), step=0.1) / 100
    w_mode = st.sidebar.number_input("WACC base (%)", value=(wacc * 100), step=0.1) / 100
    w_max = st.sidebar.number_input("WACC máximo (%)", value=(wacc * 100) + 2.0, step=0.1) / 100

st.sidebar.subheader("Variable 3: CAPEX (Triangular)")
capex_min = st.sidebar.number_input("CAPEX mínimo ($)", value=max(0.0, capex0 * 0.90), step=10000.0)
capex_mode = st.sidebar.number_input("CAPEX base ($)", value=float(capex0), step=10000.0)
capex_max = st.sidebar.number_input("CAPEX máximo ($)", value=capex0 * 1.20, step=10000.0)

st.sidebar.divider()
st.sidebar.subheader("Modo Comité de Inversión")
committee_mode = st.sidebar.checkbox("Activar Modo Comité (veredicto automático)", value=True)
max_prob_negative = st.sidebar.slider("Umbral máximo P(VAN<0)", 0.0, 1.0, 0.20, 0.01)
require_p50_positive = st.sidebar.checkbox("Exigir P50(VAN) > 0", value=True)

use_p5_floor = st.sidebar.checkbox("Exigir P5(VAN) ≥ piso", value=False)
p5_floor = st.sidebar.number_input("Piso P5(VAN) $", value=-50000, step=10000) if use_p5_floor else None

st.sidebar.caption("Tip: comité conservador → 20% y P5≥0. Comité agresivo → 30% y sin piso P5.")

def tri_ok(a, m, b):
    return a <= m <= b

@st.cache_data(show_spinner=False)
def run_monte_carlo(mc_sims, fcf_y1, n_years, terminal_growth, min_spread,
                    g_min, g_mode, g_max, w_min, w_mode, w_max,
                    capex_min, capex_mode, capex_max):
    rng = np.random.default_rng()

    g_s = rng.triangular(g_min, g_mode, g_max, mc_sims)
    w_s = rng.triangular(w_min, w_mode, w_max, mc_sims)
    capex_s = rng.triangular(capex_min, capex_mode, capex_max, mc_sims)

    yrs = np.arange(1, n_years + 1)

    fcf_paths = fcf_y1 * (1.0 + g_s)[:, None] ** (yrs[None, :] - 1)

    valid = w_s > (terminal_growth + min_spread)
    npv_s = np.full(mc_sims, np.nan)
    idx = np.where(valid)[0]

    if idx.size == 0:
        return npv_s, g_s, w_s, capex_s, idx

    fcf_valid = fcf_paths[idx, :]
    w_valid = w_s[idx]
    capex_valid = capex_s[idx]

    tv_valid = (fcf_valid[:, -1] * (1.0 + terminal_growth)) / (w_valid - terminal_growth)
    fcf_valid[:, -1] = fcf_valid[:, -1] + tv_valid

    discount = (1.0 + w_valid)[:, None] ** (yrs[None, :])
    pv = np.sum(fcf_valid / discount, axis=1)

    npv_s[idx] = pv - capex_valid
    return npv_s, g_s, w_s, capex_s, idx

def committee_verdict(prob_neg, p50, p5, max_prob_negative, require_p50_positive, use_p5_floor, p5_floor):
    checks = []
    checks.append(("P(VAN<0)", prob_neg <= max_prob_negative))
    if require_p50_positive:
        checks.append(("P50(VAN)>0", p50 > 0))
    if use_p5_floor and p5_floor is not None:
        checks.append(("P5(VAN)≥piso", p5 >= p5_floor)

        )

    passed = [ok for _, ok in checks]
    n_ok = sum(passed)
    n_total = len(passed)

    if n_ok == n_total:
        return "APROBADO ✅", "Cumple todos los criterios del comité."
    if n_ok == 0:
        return "RECHAZADO ⛔", "No cumple los criterios mínimos de riesgo/valor."

    close_to_prob = abs(prob_neg - max_prob_negative) <= 0.03
    if close_to_prob or (n_ok == n_total - 1):
        return "OBSERVADO ⚠️", "Requiere mitigaciones / mejor evidencia antes de aprobar."
    return "RECHAZADO ⛔", "Riesgo/retorno insuficiente bajo criterios actuales."

def safe_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 10:
        return np.nan
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]

def recommend_actions(prob_neg, p5, p50, capex_base, threshold):
    actions = []

    if (prob_neg <= threshold) and (p50 > 0):
        actions.append("Mantener supuestos, documentar evidencia y preparar defensa: hipótesis → fuente → indicador → decisión.")
        actions.append("Siguiente paso: pasar a validación técnica/legal y recién luego a finanzas detalladas.")
        actions.append("Asegurar trazabilidad: cada supuesto clave debe citar indicador específico (INE/BCP/MIC, etc.).")
        return actions

    if prob_neg > threshold:
        actions.append("Reducir downside: mitigar riesgos antes de aprobar (mejor evidencia de mercado, revisar costos, timing, alcance).")

    if p50 <= 0:
        actions.append("Mejorar caso base: revisar pricing/margen/volumen (mercado) o reducir CAPEX para cruzar a VAN positivo.")

    if p5 < -0.15 * capex_base:
        actions.append("Downside severo: implementar faseo (pilot → expansión) o rediseñar estructura/alcance del proyecto.")

    actions.append("Palancas típicas: (i) CAPEX (faseo / negociación / scope), (ii) Riesgo (contratos, proveedores, compliance), (iii) Mercado (validación con indicadores + pruebas piloto).")
    return actions

if enable_mc:
    st.divider()
    st.subheader("🎲 Monte Carlo: Distribución del VAN (NPV) + Veredicto Comité")

    if not tri_ok(g_min, g_mode, g_max):
        st.error("Triangular g inválida: debe cumplirse g mínimo ≤ g base ≤ g máximo.")
        st.stop()

    if not tri_ok(w_min, w_mode, w_max):
        st.error("Triangular WACC inválida: debe cumplirse WACC mínimo ≤ base ≤ máximo.")
        st.stop()

    if not tri_ok(capex_min, capex_mode, capex_max):
        st.error("Triangular CAPEX inválida: debe cumplirse CAPEX mínimo ≤ base ≤ máximo.")
        st.stop()

    npv_s, g_s, w_s, capex_s, idx = run_monte_carlo(
        mc_sims, fcf_y1, n_years, terminal_growth, MIN_SPREAD,
        g_min, g_mode, g_max, w_min, w_mode, w_max,
        capex_min, capex_mode, capex_max
    )

    if idx.size == 0:
        st.error("Ninguna simulación cumple WACC > g∞ + 0.5%. Ajustá WACC o g∞.")
        st.stop()

    prob_neg = np.nanmean(npv_s < 0)
    p5, p50, p95 = np.nanpercentile(npv_s, [5, 50, 95])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Simulaciones", f"{mc_sims:,}")
    m2.metric("P(VAN < 0)", f"{prob_neg:.1%}")
    m3.metric("P5(VAN)", f"${p5:,.0f}")
    m4.metric("P50(VAN)", f"${p50:,.0f}")
    m5.metric("P95(VAN)", f"${p95:,.0f}")

    # Veredicto comité + checklist
    verdict = None
    if committee_mode:
        verdict, rationale = committee_verdict(
            prob_neg, p50, p5,
            max_prob_negative, require_p50_positive,
            use_p5_floor, p5_floor
        )
        st.subheader(f"🧑‍⚖️ Veredicto Comité: {verdict}")
        st.write(rationale)

        st.markdown("**Criterios aplicados:**")
        crit_lines = []
        crit_lines.append(f"- P(VAN<0) ≤ {max_prob_negative:.0%}  → **{'OK' if prob_neg <= max_prob_negative else 'NO'}**")
        if require_p50_positive:
            crit_lines.append(f"- P50(VAN) > 0  → **{'OK' if p50 > 0 else 'NO'}**")
        if use_p5_floor and p5_floor is not None:
            crit_lines.append(f"- P5(VAN) ≥ {p5_floor:,.0f}  → **{'OK' if p5 >= p5_floor else 'NO'}**")
        st.markdown("\n".join(crit_lines))

    # Histograma
    fig_mc = px.histogram(
        x=npv_s[~np.isnan(npv_s)],
        nbins=50,
        labels={"x": "VAN (NPV)"},
        title="Distribución del VAN (Monte Carlo)"
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # Drivers (muestra)
    st.caption("Drivers (muestra): relación entre VAN y variables simuladas")
    rng = np.random.default_rng()
    sample_n = min(2000, idx.size)
    sample_idx = rng.choice(idx, size=sample_n, replace=False)

    df_scatter = {
        "NPV": npv_s[sample_idx],
        "g": g_s[sample_idx],
        "WACC": w_s[sample_idx],
        "CAPEX": capex_s[sample_idx]
    }

    fig_sc = px.scatter(
        df_scatter,
        x="WACC",
        y="NPV",
        hover_data=["g", "CAPEX"],
        title="NPV vs WACC (muestra Monte Carlo)"
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # ------------------------------------------------------------
    # TARJETA: ACCIÓN RECOMENDADA
    # ------------------------------------------------------------
    st.markdown("### 🧭 Acción recomendada")

    # Interpretación de drivers por correlación (aprox)
    corr_g = safe_corr(df_scatter["NPV"], df_scatter["g"])
    corr_w = safe_corr(df_scatter["NPV"], df_scatter["WACC"])
    corr_c = safe_corr(df_scatter["NPV"], df_scatter["CAPEX"])

    driver_notes = []
    if not np.isnan(corr_g) and abs(corr_g) > 0.2:
        driver_notes.append(f"Driver: crecimiento g (corr≈{corr_g:+.2f}).")
    if not np.isnan(corr_w) and abs(corr_w) > 0.2:
        driver_notes.append(f"Driver: WACC (corr≈{corr_w:+.2f}).")
    if not np.isnan(corr_c) and abs(corr_c) > 0.2:
        driver_notes.append(f"Driver: CAPEX (corr≈{corr_c:+.2f}).")

    # Mensaje principal según veredicto
    if committee_mode and verdict is not None:
        if "APROBADO" in verdict:
            st.success("Acción sugerida: avanzar a la siguiente etapa (técnico/legal/finanzas) con evidencia documentada.")
        elif "OBSERVADO" in verdict:
            st.warning("Acción sugerida: mitigar riesgos y fortalecer evidencia antes de pasar de etapa.")
        else:
            st.error("Acción sugerida: replantear supuestos/alcance o rediseñar el proyecto antes de insistir con el DCF.")
    else:
        # si no hay modo comité, igual damos guía
        if prob_neg <= max_prob_negative and p50 > 0:
            st.success("Lectura rápida: el perfil de riesgo parece aceptable. Documentá evidencia y prepará defensa.")
        else:
            st.warning("Lectura rápida: el perfil de riesgo requiere mitigación y mejor evidencia.")

    recs = recommend_actions(
        prob_neg=prob_neg,
        p5=p5,
        p50=p50,
        capex_base=capex_mode,
        threshold=max_prob_negative
    )

    for r in recs:
        st.write(f"• {r}")

    if driver_notes:
        st.caption("Notas de drivers (muestra, correlación aprox.): " + " ".join(driver_notes))
    else:
        st.caption("Notas de drivers: sin señal fuerte en la muestra (o muestra pequeña).")

    st.info(
        "Interpretación Comité: el VAN es una distribución. "
        "La decisión debe mirar P(VAN<0) + percentiles (P5/P50/P95). "
        "Si el downside es alto, proponé mitigaciones (CAPEX, timing, mercado, costos) antes de aprobar."
    )

# ============================================================
# Proxy tipo Altman (pedagógico)
# ============================================================
st.sidebar.divider()
st.sidebar.subheader("Bono: Salud financiera (Proxy tipo Altman)")

working_cap = st.sidebar.number_input("Capital de Trabajo (WC)", value=50000, step=5000)
retained_earnings = st.sidebar.number_input("Utilidades Retenidas (RE)", value=30000, step=5000)
ebit = st.sidebar.number_input("EBIT", value=80000, step=5000)
total_assets = st.sidebar.number_input("Activos Totales (TA)", value=1000000, step=10000)
use_sales = st.sidebar.checkbox("Usar Ventas (Sales) en proxy", value=True)
sales = st.sidebar.number_input("Ventas (Sales)", value=900000, step=10000) if use_sales else None

st.divider()
st.subheader("🩺 Salud Financiera (Proxy tipo Altman - pedagógico)")

if total_assets <= 0 or deuda <= 0:
    st.info("Proxy no disponible: verificá Activos Totales (TA) > 0 y Deuda (D) > 0.")
else:
    x1 = working_cap / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = equity / deuda
    x5 = (sales / total_assets) if (use_sales and sales is not None and total_assets > 0) else (np.mean(cash_flows) / total_assets)

    z_proxy = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (1.0 * x5)

    if z_proxy > 2.99:
        st.success(f"Z-Proxy: {z_proxy:.2f} (Zona Segura - interpretación pedagógica)")
    elif z_proxy > 1.81:
        st.warning(f"Z-Proxy: {z_proxy:.2f} (Zona Precaución - interpretación pedagógica)")
    else:
        st.error(f"Z-Proxy: {z_proxy:.2f} (Zona Riesgo - interpretación pedagógica)")

st.caption(
    "Notas: (1) VAN/TIR corresponden a DCF de Enterprise Value (proyecto) con WACC y CAPEX explícito. "
    "(2) La estructura de capital se usa para WACC (no es la inversión inicial). "
    "(3) El Z-Proxy es una aproximación pedagógica."
)
