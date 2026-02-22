import streamlit as st
import numpy as np
import numpy_financial as npf
import plotly.express as px
from dataclasses import dataclass
from datetime import date
import io
import urllib.request

# ============================================================
# ReportLab (PDF) - Import protegido (no rompe la app si falta)
# ============================================================
REPORTLAB_OK = True
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
except Exception:
    REPORTLAB_OK = False

MIN_SPREAD = 0.005  # Guardrail: WACC > g∞ + 0.5%


# ============================================================
# UI base
# ============================================================
st.set_page_config(page_title="ValuationApp – USIL MBA (Paraguay)", layout="wide")
st.title("📊 ValuationApp – Evaluación de Proyectos (Comité Académico MBA – USIL)")
st.caption(
    "Marco: DCF + evaluación probabilística (Monte Carlo). "
    "Incluye explicaciones académicas, criterios de comité e informe ejecutivo (2 páginas)."
)


# ============================================================
# Helpers
# ============================================================
def money(x: float, ccy: str = "USD") -> str:
    # Formato simple por moneda. Mantengo $ por defecto por consistencia visual.
    # Si querés, podemos cambiar a "Gs." cuando ccy=PYG.
    return f"${x:,.0f}"

def pct(x: float) -> str:
    return f"{x:.2%}"

def tri_ok(a, m, b) -> bool:
    return a <= m <= b

def safe_irr(x):
    if x is None or np.isnan(x) or np.isinf(x):
        return None
    # En flujos no convencionales puede ser múltiple o no existir; filtro conservador
    if x < -1.0 or x > 2.0:
        return None
    return float(x)

def safe_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 10:
        return np.nan
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def badge(verdict: str) -> str:
    return {"APROBADO": "✅", "OBSERVADO": "⚠️", "RECHAZADO": "⛔"}.get(verdict, "—")


# ============================================================
# Comité: veredicto y recomendaciones
# ============================================================
def committee_verdict(prob_neg, p50, p5, max_prob_negative, require_p50_positive, use_p5_floor, p5_floor):
    checks = []
    checks.append(("P(VAN<0)", prob_neg <= max_prob_negative))
    if require_p50_positive:
        checks.append(("P50(VAN) > 0", p50 > 0))
    if use_p5_floor and (p5_floor is not None):
        checks.append(("P5(VAN) ≥ piso", p5 >= p5_floor))

    n_ok = sum(ok for _, ok in checks)
    n_total = len(checks)

    if n_ok == n_total:
        return (
            "APROBADO",
            "El proyecto presenta un perfil riesgo–retorno compatible con los criterios establecidos, "
            "evidenciando una expectativa favorable de creación de valor económico."
        )

    if n_ok == 0:
        return (
            "RECHAZADO",
            "El proyecto no cumple los criterios mínimos definidos. "
            "El perfil riesgo–retorno es incompatible con una decisión favorable en su estado actual."
        )

    return (
        "OBSERVADO",
        "El proyecto presenta potencial de creación de valor; sin embargo, evidencia vulnerabilidades "
        "que justifican fortalecer supuestos, evidencia y mitigaciones antes de su eventual aprobación."
    )

def recommended_actions(prob_neg, p5, p50, capex_base, threshold, driver_focus: str | None):
    actions: list[str] = []

    if (prob_neg <= threshold) and (p50 > 0):
        actions.append(
            "Avanzar a la siguiente etapa, consolidando la trazabilidad de supuestos mediante evidencia empírica verificable "
            "(fuentes secundarias y levantamiento primario selectivo)."
        )
        actions.append(
            "Establecer hitos de control (control formativo) para validar supuestos críticos antes de comprometer capital adicional."
        )
        actions.append(
            "Documentar explícitamente supuestos clave (demanda, precios, costos, inversión, tasa) y su justificación."
        )
        if driver_focus:
            actions.append(
                f"Priorizar evidencia y mitigaciones sobre determinantes con sensibilidad relevante: {driver_focus}."
            )
        return actions

    actions.append(
        "Fortalecer supuestos fundamentales del caso antes de una aprobación, dado el perfil de riesgo evidenciado."
    )

    if prob_neg > threshold:
        actions.append(
            "Reducir incertidumbre en variables críticas con evidencia adicional y/o rediseño de supuestos, "
            "dado que P(VAN<0) resulta significativa respecto del umbral definido."
        )

    if p50 <= 0:
        actions.append(
            "Revisar modelo de ingresos/costos y estructura de inversión para mejorar la robustez del caso base (P50 no favorable)."
        )

    if p5 < -0.15 * capex_base:
        actions.append(
            "Considerar una estrategia por fases (piloto → escalamiento) para reducir exposición inicial, "
            "dado un downside material en P5."
        )

    actions.append(
        "Líneas de acción: (i) racionalización/faseo del CAPEX, (ii) validación de demanda con indicadores y evidencia primaria, "
        "(iii) optimización de márgenes/costos, (iv) mitigación contractual y de ejecución."
    )

    if driver_focus:
        actions.append(
            f"Tratar como supuestos críticos aquellos vinculados a: {driver_focus}, priorizando evidencia y mitigación."
        )

    return actions


# ============================================================
# Monte Carlo
# ============================================================
@st.cache_data(show_spinner=False)
def run_monte_carlo(
    sims: int,
    fcf_y1: float,
    n_years: int,
    g_inf: float,
    min_spread: float,
    g_min: float, g_mode: float, g_max: float,
    w_min: float, w_mode: float, w_max: float,
    capex_min: float, capex_mode: float, capex_max: float
):
    rng = np.random.default_rng()

    g_s = rng.triangular(g_min, g_mode, g_max, sims)
    w_s = rng.triangular(w_min, w_mode, w_max, sims)
    capex_s = rng.triangular(capex_min, capex_mode, capex_max, sims)

    yrs = np.arange(1, n_years + 1)
    fcf_paths = fcf_y1 * (1.0 + g_s)[:, None] ** (yrs[None, :] - 1)

    # Consistencia para TV: WACC > g∞ + spread
    valid = w_s > (g_inf + min_spread)

    npv_s = np.full(sims, np.nan)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return npv_s, g_s, w_s, capex_s, idx

    fcf_valid = fcf_paths[idx, :]
    w_valid = w_s[idx]
    capex_valid = capex_s[idx]

    tv_valid = (fcf_valid[:, -1] * (1.0 + g_inf)) / (w_valid - g_inf)
    fcf_valid[:, -1] += tv_valid

    discount = (1.0 + w_valid)[:, None] ** (yrs[None, :])
    pv = np.sum(fcf_valid / discount, axis=1)

    npv_s[idx] = pv - capex_valid
    return npv_s, g_s, w_s, capex_s, idx


# ============================================================
# Informe ejecutivo (texto + PDF)
# ============================================================
@dataclass
class ExecReport:
    institution: str
    program: str
    course: str
    project: str
    responsible: str

    currency: str
    basis: str
    d_e_basis: str
    crp_approach: str

    capex0: float
    wacc: float
    base_npv: float
    base_irr: float | None

    sims: int
    prob_neg: float
    p5: float
    p50: float
    p95: float

    verdict: str
    rationale: str
    criteria_lines: list[str]
    driver_focus: str | None
    actions: list[str]
    limitations: list[str]


def build_executive_text(r: ExecReport) -> str:
    irr_text = pct(r.base_irr) if r.base_irr is not None else "N/A (posible no unicidad/no existencia)"

    lines = []
    lines.append(r.institution)
    lines.append(r.program)
    lines.append(r.course)
    lines.append(f"Fecha: {date.today().isoformat()}")
    lines.append("")
    lines.append("INFORME EJECUTIVO (2 PÁGINAS) – EVALUACIÓN DE PROYECTO")
    lines.append(f"Proyecto: {r.project}")
    lines.append(f"Responsable: {r.responsible}")
    lines.append("")
    lines.append(f"1. Síntesis ejecutiva – Veredicto: {r.verdict} {badge(r.verdict)}")
    lines.append(r.rationale)
    lines.append("")
    lines.append("2. Indicadores principales")
    lines.append(f"- CAPEX (Año 0): {money(r.capex0, r.currency)} ({r.currency})")
    lines.append(f"- WACC (tasa de descuento): {pct(r.wacc)}")
    lines.append(f"- VAN base (determinístico): {money(r.base_npv, r.currency)}")
    lines.append(f"- TIR base (determinística): {irr_text}")
    lines.append(f"- P(VAN<0): {r.prob_neg:.1%}")
    lines.append(f"- P50 (resultado central): {money(r.p50, r.currency)}")
    lines.append(f"- P5 (escenario adverso plausible): {money(r.p5, r.currency)}")
    lines.append(f"- P95 (escenario favorable): {money(r.p95, r.currency)}")
    lines.append("")
    lines.append("3. Lectura probabilística del riesgo (registro no técnico)")
    lines.append(
        f"El análisis Monte Carlo caracteriza la distribución del VAN bajo incertidumbre razonable. "
        f"El resultado central (P50) se estima en {money(r.p50, r.currency)}, mientras que el escenario adverso plausible (P5) "
        f"alcanza {money(r.p5, r.currency)}. La probabilidad de destrucción de valor P(VAN<0) se estima en {r.prob_neg:.1%}."
    )
    if r.driver_focus:
        lines.append(
            f"Se observa sensibilidad relevante asociada a: {r.driver_focus} "
            f"(señal orientativa para priorización de supuestos críticos; no implica causalidad)."
        )

    lines.append("")
    lines.append("4. Criterios del Comité")
    if r.criteria_lines:
        for cl in r.criteria_lines:
            lines.append(f"- {cl}")
    else:
        lines.append("- Modo comité desactivado; sin criterios automáticos.")
    lines.append("")
    lines.append("5. Supuestos y consistencia metodológica")
    lines.append(f"- Moneda: {r.currency}")
    lines.append(f"- Base de medición (tasa y flujos): {r.basis}")
    lines.append(f"- Estructura D/E utilizada para WACC: {r.d_e_basis}")
    lines.append(f"- Riesgo país (CRP): {r.crp_approach}")
    lines.append(
        "- Modelo de FCF: proyección parsimoniosa con crecimiento constante (apropiado para discusión académica); "
        "en evaluación profesional se recomienda desagregar por drivers (ingresos, margen, reinversión y capital de trabajo)."
    )
    lines.append("")
    lines.append("6. Recomendación y plan de acción")
    for a in r.actions:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("7. Limitaciones (declaración académica)")
    for l in r.limitations:
        lines.append(f"- {l}")

    return "\n".join(lines)


def _load_logo_reader(uploaded_file, url: str | None):
    if not REPORTLAB_OK:
        return None
    try:
        if uploaded_file is not None:
            return ImageReader(uploaded_file)
        if url:
            with urllib.request.urlopen(url) as resp:
                data = resp.read()
            return ImageReader(io.BytesIO(data))
    except Exception:
        return None
    return None


def _wrap_lines(text: str, max_chars: int):
    words = text.split()
    out = []
    line = ""
    for w in words:
        if len(line) + len(w) + 1 <= max_chars:
            line = (line + " " + w).strip()
        else:
            out.append(line)
            line = w
    if line:
        out.append(line)
    return out


def generate_pdf_2pages(r: ExecReport, logo_reader=None) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab no está disponible en este entorno. Revise requirements.txt.")

    tmp = io.BytesIO()
    c = canvas.Canvas(tmp, pagesize=letter)
    width, height = letter

    left = 0.75 * inch
    top = height - 0.75 * inch
    leading = 12
    max_chars = 105

    def header(y, page_no):
        if logo_reader is not None:
            c.drawImage(logo_reader, left, y - 40, width=120, height=40, mask="auto")
            c.setFont("Helvetica-Bold", 11.5)
            c.drawString(left + 130, y - 12, r.institution)
            c.setFont("Helvetica", 10.5)
            c.drawString(left + 130, y - 28, r.program)
            c.drawString(left + 130, y - 42, r.course)
        else:
            c.setFont("Helvetica-Bold", 11.5)
            c.drawString(left, y, r.institution)
            c.setFont("Helvetica", 10.5)
            c.drawString(left, y - 14, r.program)
            c.drawString(left, y - 28, r.course)

        c.setFont("Helvetica", 9.5)
        c.drawRightString(width - left, y - 14, f"Fecha: {date.today().isoformat()}")
        c.drawRightString(width - left, y - 28, f"Página {page_no} de 2")
        return y - 60

    def h1(y, text):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(left, y, text)
        return y - 18

    def h2(y, text):
        c.setFont("Helvetica-Bold", 11.5)
        c.drawString(left, y, text)
        return y - 16

    def p(y, text):
        c.setFont("Helvetica", 10.5)
        for line in _wrap_lines(text, max_chars):
            c.drawString(left, y, line)
            y -= leading
        return y

    def bullets(y, items):
        c.setFont("Helvetica", 10.5)
        for it in items:
            for line in _wrap_lines(f"• {it}", max_chars):
                c.drawString(left, y, line)
                y -= leading
        return y

    # Page 1
    y = header(top, 1)
    y = h1(y, "Informe Ejecutivo – Evaluación de Proyecto (2 páginas)")
    y = p(y, f"Proyecto: {r.project}  |  Responsable: {r.responsible}")
    y -= 6

    y = h2(y, f"1. Síntesis ejecutiva – Veredicto: {r.verdict} {badge(r.verdict)}")
    y = p(y, r.rationale)
    y -= 6

    irr_text = pct(r.base_irr) if r.base_irr is not None else "N/A (posible no unicidad/no existencia)"
    y = h2(y, "2. Indicadores principales")
    y = bullets(y, [
        f"CAPEX (Año 0): {money(r.capex0, r.currency)} ({r.currency})",
        f"WACC (tasa de descuento): {pct(r.wacc)}",
        f"VAN base (determinístico): {money(r.base_npv, r.currency)}",
        f"TIR base (determinística): {irr_text}",
        f"Monte Carlo: {r.sims:,} simulaciones | P(VAN<0) {r.prob_neg:.1%}",
        f"P50: {money(r.p50, r.currency)} | P5: {money(r.p5, r.currency)} | P95: {money(r.p95, r.currency)}",
    ])
    y -= 6

    y = h2(y, "3. Lectura probabilística del riesgo (registro no técnico)")
    y = p(y,
          f"El análisis Monte Carlo caracteriza la distribución del VAN bajo incertidumbre razonable. "
          f"El resultado central (P50) se estima en {money(r.p50, r.currency)}, mientras que el escenario adverso plausible (P5) "
          f"alcanza {money(r.p5, r.currency)}. La probabilidad de destrucción de valor P(VAN<0) se estima en {r.prob_neg:.1%}."
          )
    if r.driver_focus:
        y = p(y,
              f"Se observa sensibilidad relevante asociada a: {r.driver_focus} "
              f"(señal orientativa para priorización de supuestos críticos; no implica causalidad)."
              )

    y -= 4
    y = h2(y, "4. Criterios del Comité")
    if r.criteria_lines:
        y = bullets(y, r.criteria_lines)
    else:
        y = bullets(y, ["Modo comité desactivado; sin criterios automáticos."])

    c.showPage()

    # Page 2
    y = header(top, 2)
    y = h1(y, "Informe Ejecutivo – Continuación")

    y = h2(y, "5. Supuestos y consistencia metodológica")
    y = bullets(y, [
        f"Moneda: {r.currency}.",
        f"Base de medición (tasa y flujos): {r.basis}.",
        f"Estructura D/E para WACC: {r.d_e_basis}.",
        f"Riesgo país (CRP): {r.crp_approach}.",
        "Modelo FCF: proyección parsimoniosa con crecimiento constante; se recomienda desagregar por drivers en evaluación profesional.",
    ])
    y -= 6

    y = h2(y, "6. Recomendación y plan de acción")
    y = bullets(y, r.actions[:6])
    y -= 6

    y = h2(y, "7. Limitaciones (declaración académica)")
    y = bullets(y, r.limitations[:6])
    y -= 10

    c.setFont("Helvetica-Oblique", 9.5)
    y = p(y,
          "Nota: Este informe ejecutivo prioriza lectura de comité. Los resultados dependen de supuestos (mercado, inversión, tasa). "
          "Se recomienda sostener supuestos críticos con evidencia y establecer mitigaciones antes de comprometer capital."
          )

    c.save()
    pdf_bytes = tmp.getvalue()
    return pdf_bytes


# ============================================================
# Sidebar – Encabezado USIL + Paraguay context
# ============================================================
st.sidebar.header("🏛️ Encabezado institucional (USIL – MBA)")
institution = st.sidebar.text_input("Institución", "Universidad San Ignacio de Loyola (USIL)")
program = st.sidebar.text_input("Programa", "Maestría en Administración de Negocios (MBA)")
course = st.sidebar.text_input("Curso / Módulo", "Proyectos de Inversión / Valuation")

st.sidebar.divider()
st.sidebar.header("🧩 Identificación")
project = st.sidebar.text_input("Proyecto", "Proyecto")
responsible = st.sidebar.text_input("Responsable", "Docente: Jorge Rojas")

st.sidebar.divider()
st.sidebar.header("🔎 Consistencia (Paraguay)")
currency = st.sidebar.selectbox("Moneda del modelo", ["USD", "PYG"], index=0)
basis = st.sidebar.selectbox("Base de medición (tasa y flujos)", ["Nominal", "Real"], index=0)
d_e_basis = st.sidebar.selectbox(
    "Base D/E para WACC (Paraguay)",
    ["Valores contables (práctica frecuente en PY)", "Valores de mercado (si disponible)", "Mixto/estimado"],
    index=0
)

st.sidebar.divider()
st.sidebar.header("🖼️ Logo en PDF (opcional)")
logo_file = st.sidebar.file_uploader("Subir logo (PNG/JPG)", type=["png", "jpg", "jpeg"])
logo_url = st.sidebar.text_input("o URL directa del logo (.png/.jpg)", value="").strip()
logo_reader = _load_logo_reader(logo_file, logo_url if logo_url else None)

st.sidebar.caption(
    "Para PDF institucional: ideal PNG con buena resolución. "
    "Si reportlab no está instalado, el PDF se deshabilita automáticamente."
)

# ============================================================
# Inputs financieros
# ============================================================
st.sidebar.divider()
st.sidebar.header("0) Inversión inicial")
capex0 = st.sidebar.number_input("CAPEX Año 0", value=500000.0, step=10000.0, min_value=1.0)

st.sidebar.divider()
st.sidebar.header("1) CAPM / WACC")
rf = st.sidebar.number_input("Rf (%)", value=4.5, step=0.1) / 100
erp = st.sidebar.number_input("ERP (%)", value=5.5, step=0.1) / 100

use_crp = st.sidebar.checkbox("Incluir Riesgo País (CRP) en Ke", value=True)
crp = st.sidebar.number_input("CRP (%)", value=2.0, step=0.1) / 100 if use_crp else 0.0

beta_u = st.sidebar.number_input("βU (desapalancada)", value=0.90, step=0.05)
tax_rate = st.sidebar.number_input("Impuesto T (%)", value=10.0, step=0.5) / 100

st.sidebar.divider()
st.sidebar.header("2) Estructura de capital (para WACC)")
deuda = st.sidebar.number_input("Deuda D", value=400000.0, step=10000.0, min_value=0.0)
equity = st.sidebar.number_input("Capital propio E", value=600000.0, step=10000.0, min_value=1.0)
kd = st.sidebar.number_input("Kd (%)", value=7.0, step=0.1) / 100

st.sidebar.divider()
st.sidebar.header("3) Flujos (FCF)")
n_years = st.sidebar.slider("Años de proyección", 1, 15, 5)
fcf_y1 = st.sidebar.number_input("FCF Año 1", value=100000.0, step=5000.0)
g = st.sidebar.number_input("g (%) – crecimiento explícito", value=5.0, step=0.1) / 100
g_inf = st.sidebar.number_input("g∞ (%) – crecimiento perpetuo", value=2.0, step=0.1) / 100


# ============================================================
# Cálculos base (DCF + WACC)
# ============================================================
beta_l = beta_u * (1 + (1 - tax_rate) * (deuda / equity))  # Hamada
ke = rf + beta_l * erp + (crp if use_crp else 0.0)

total = deuda + equity
wacc = (equity / total) * ke + (deuda / total) * kd * (1 - tax_rate)

if wacc <= g_inf + MIN_SPREAD:
    st.error(
        "Condición de consistencia: WACC debe ser mayor que g∞ por al menos 0.5%. "
        "Ajuste g∞ o revise WACC (Ke, Kd, estructura D/E)."
    )
    st.stop()

years = np.arange(1, n_years + 1)
cash_flows = fcf_y1 * (1 + g) ** (years - 1)

# Valor terminal (Gordon–Shapiro)
tv = (cash_flows[-1] * (1 + g_inf)) / (wacc - g_inf)

flows = cash_flows.copy()
flows[-1] += tv

base_npv = float(npf.npv(wacc, [-capex0] + flows.tolist()))
base_irr = safe_irr(npf.irr([-capex0] + flows.tolist()))


# ============================================================
# UI: Resumen base
# ============================================================
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("βU", f"{beta_u:.2f}")
k2.metric("βL", f"{beta_l:.2f}")
k3.metric("WACC", pct(wacc))
k4.metric("VAN (base)", money(base_npv, currency))
k5.metric("TIR (base)", pct(base_irr) if base_irr is not None else "N/A")

fig_base = px.bar(x=years, y=cash_flows, labels={"x": "Año", "y": "FCF"}, title="Flujos proyectados (FCF) – Período explícito")
st.plotly_chart(fig_base, use_container_width=True)
st.caption(f"Valor Terminal (TV) incorporado en el último año: {money(float(tv), currency)}")

st.divider()


# ============================================================
# Explicaciones dentro de la app (lo que pediste)
# ============================================================
with st.expander("📚 Explicaciones metodológicas (MBA – adaptado a Paraguay)", expanded=False):
    st.markdown(
        """
### 1) g (crecimiento explícito) vs g∞ (crecimiento perpetuo)

- **g (crecimiento explícito):** tasa de crecimiento aplicada durante el **horizonte proyectado** (años 1 a N).  
  Representa la fase activa del proyecto (expansión, penetración de mercado, mejoras operativas).

- **g∞ (crecimiento perpetuo):** tasa de crecimiento sostenible a **largo plazo**, utilizada para calcular el **Valor Terminal**.  
  Representa una fase de madurez (crecimiento estructural compatible con la economía).

✅ Condición esencial: **g∞ debe ser menor que WACC** (y normalmente menor que g).

---

### 2) DCF (Discounted Cash Flow / Flujo de Caja Descontado)
El método DCF:
1) **Proyecta** flujos futuros (FCF)  
2) **Descuenta** esos flujos a valor presente con el **WACC**  

Interpretación:
- **VAN > 0:** el proyecto **crea valor** bajo supuestos del modelo.
- **VAN < 0:** el proyecto **destruye valor** bajo esos supuestos.

---

### 3) βU y βL (beta desapalancada vs apalancada)
- **βU (desapalancada):** riesgo operativo de la industria **sin deuda**.
- **βL (apalancada):** riesgo del accionista **considerando deuda** (efecto del apalancamiento).

Se utiliza (Hamada):
\[
\\beta_L = \\beta_U \\cdot \\left(1 + (1-T)\\frac{D}{E}\\right)
\]

Más deuda (D/E) → mayor βL → mayor Ke.

---

### 4) P5, P50, P95 (percentiles del VAN en Monte Carlo)
- **P50:** resultado central (50% de escenarios mejores, 50% peores).  
- **P5:** escenario adverso plausible (solo 5% de escenarios peores).  
- **P95:** escenario favorable plausible (solo 5% mejores).  

Además:
- **P(VAN<0):** probabilidad estimada de destrucción de valor.

---

### 5) ¿Qué son “flujos no convencionales” y por qué importa la TIR?
Un flujo “convencional” cambia de signo una sola vez:  
- Año 0 negativo (inversión) y luego flujos positivos.

Un flujo “no convencional” cambia de signo más de una vez, por ejemplo:  
- inversión inicial, luego flujos positivos, luego un gran reintegro/inversión posterior, etc.

Esto puede provocar:
- **múltiples TIR** o
- **TIR no definida**.

Por eso, en análisis profesional y académico se recomienda **priorizar el VAN** como criterio principal.
"""
    )


# ============================================================
# NUEVA SECCIÓN: Cómo estimar parámetros en Paraguay (paso a paso)
# ============================================================
with st.expander("🇵🇾 Cómo estimar parámetros correctamente en Paraguay (paso a paso)", expanded=True):
    st.markdown(
        """
Esta guía está diseñada para el contexto paraguayo, donde el mercado de capitales puede presentar **baja profundidad y liquidez**.  
El objetivo es mantener el modelo **académicamente defendible** y **prácticamente aplicable**.

## Paso 1: Definir moneda y consistencia (Nominal vs Real)
- Si tus flujos (FCF) están en **USD nominal**, tu WACC debe ser **USD nominal**.
- Si los flujos están en **PYG nominal**, tu WACC debe ser **PYG nominal**.

**Recomendación práctica en Paraguay:**  
- Usar USD puede simplificar comparabilidad y estabilidad, especialmente para proyectos con exposición a importaciones o precios referenciados.

## Paso 2: Tasa libre de riesgo (Rf)
Buenas prácticas:
- En USD: usar un referente soberano de alta calidad (p.ej., bonos del Tesoro de EE. UU.) como proxy de Rf.
- En PYG: en la práctica se utilizan referencias locales, pero la “pureza” de Rf puede ser discutible; por ello es común trabajar en USD y luego ajustar.

**Consejo de comité:** declarar la fuente utilizada (en el informe) y justificarla.

## Paso 3: Prima de riesgo de mercado (ERP)
- ERP refleja el exceso de retorno esperado del mercado sobre la tasa libre de riesgo.
- Para Paraguay, se suele utilizar ERP de mercados desarrollados como base + ajustes por riesgo país o emergente (enfoque aplicado).

## Paso 4: Riesgo país (CRP)
En Paraguay es **razonable** incluir un CRP si:
- existe exposición a shocks macro,
- la generación de caja depende del contexto soberano,
- el costo de financiamiento refleja riesgo país.

En tu app, el CRP se incorpora como prima adicional en Ke (enfoque aplicado).  
✅ Es defendible, siempre que se declare como **enfoque metodológico**.

## Paso 5: Beta (βU y βL)
Paraguay tiene pocas empresas listadas y beta local observable es limitada. En la práctica:
1) se toma **βU de industria** (comparables internacionales),
2) se apalanca con D/E para obtener **βL**.

**Sugerencia:** usar comparables (misma industria) y justificar selección.

## Paso 6: Deuda (D) y Equity (E) en Paraguay: ¿valores de mercado?
- **Equity (mercado):** generalmente NO observable si la empresa no cotiza.  
  Alternativas defendibles:
  - valor estimado por DCF,
  - valor contable ajustado,
  - múltiplos de comparables (si hay información).

- **Deuda (mercado):** muchas veces tampoco cotiza.  
  Alternativas defendibles:
  - saldo financiero (book value),
  - valor presente de deuda (si hay tasa y cronograma),
  - aproximación por tasa de mercado para deuda similar.

✅ Por eso en Paraguay es frecuente usar **valores contables** o **mixtos** y declararlo como simplificación.

## Paso 7: g (crecimiento explícito) y g∞ (perpetuo)
- **g:** crecimiento del horizonte explícito (puede ser mayor, refleja expansión).
- **g∞:** crecimiento sostenible de largo plazo (normalmente menor).

Rangos de referencia (orientativos):
- En USD: g∞ suele ubicarse en el orden de **1.5%–3%**.
- En PYG: g∞ suele incluir inflación esperada; típicamente **3%–6%** según el escenario.

✅ Regla de consistencia: **WACC > g∞**.

## Paso 8: Calidad de evidencia (lo que mira un comité)
Para cada supuesto crítico, documentar:
- hipótesis → fuente → indicador → impacto en FCF/WACC

Ejemplos de evidencia en Paraguay:
- reportes sectoriales (públicos/privados),
- estadísticas oficiales y regulatorias,
- encuestas/entrevistas cortas (levantamiento primario),
- precios observables (competencia, marketplaces, distribuidores).
"""
    )

    st.info(
        "Tip de comité: si el mercado no ofrece valores de mercado observables (deuda/equity), "
        "no es un 'error'; es una restricción de contexto. Lo importante es declarar la aproximación, "
        "sustentarla con evidencia y hacer sensibilidad/Monte Carlo."
    )


# ============================================================
# Monte Carlo + Comité
# ============================================================
st.header("🎲 Monte Carlo + Comité (interpretación ejecutiva)")

enable_mc = st.checkbox("Activar Monte Carlo", value=True)
sims = st.slider("Simulaciones", 1000, 50000, 10000, 1000)

st.subheader("Rangos (triangular) – incertidumbre razonable")
st.caption("Mínimo = adverso, Base = más probable, Máximo = favorable.")

# Rangos
g_min = st.number_input("g mínimo (%) – adverso", value=max(-5.0, g * 100 - 4.0), step=0.1) / 100
g_mode = st.number_input("g base (%) – más probable", value=g * 100, step=0.1) / 100
g_max = st.number_input("g máximo (%) – favorable", value=g * 100 + 4.0, step=0.1) / 100

auto_wacc = st.checkbox("Auto WACC (usar WACC calculado ±2%)", value=True)
if auto_wacc:
    w_min = max(0.001, wacc - 0.02)
    w_mode = wacc
    w_max = wacc + 0.02
    st.caption(f"Auto WACC: mínimo={pct(w_min)}, base={pct(w_mode)}, máximo={pct(w_max)}")
else:
    w_min = st.number_input("WACC mínimo (%)", value=max(0.1, wacc * 100 - 2.0), step=0.1) / 100
    w_mode = st.number_input("WACC base (%)", value=wacc * 100, step=0.1) / 100
    w_max = st.number_input("WACC máximo (%)", value=wacc * 100 + 2.0, step=0.1) / 100

capex_min = st.number_input("CAPEX mínimo – favorable", value=max(0.0, capex0 * 0.90), step=10000.0)
capex_mode = st.number_input("CAPEX base – más probable", value=float(capex0), step=10000.0)
capex_max = st.number_input("CAPEX máximo – adverso", value=capex0 * 1.20, step=10000.0)

# Comité
st.subheader("Criterios del Comité")
committee_mode = st.checkbox("Activar veredicto automático", value=True)
max_prob_negative = st.slider("Umbral máximo aceptable P(VAN<0)", 0.0, 1.0, 0.20, 0.01)
require_p50_positive = st.checkbox("Exigir P50(VAN) > 0", value=True)
use_p5_floor = st.checkbox("Exigir P5(VAN) ≥ piso", value=False)
p5_floor = st.number_input("Piso P5(VAN)", value=-50000.0, step=10000.0) if use_p5_floor else None


# Ejecutar MC
if enable_mc:
    if not tri_ok(g_min, g_mode, g_max):
        st.error("Rango inválido para g: debe cumplirse mínimo ≤ base ≤ máximo.")
        st.stop()
    if not tri_ok(w_min, w_mode, w_max):
        st.error("Rango inválido para WACC: debe cumplirse mínimo ≤ base ≤ máximo.")
        st.stop()
    if not tri_ok(capex_min, capex_mode, capex_max):
        st.error("Rango inválido para CAPEX: debe cumplirse mínimo ≤ base ≤ máximo.")
        st.stop()

    npv_s, g_s, w_s, capex_s, idx = run_monte_carlo(
        sims=int(sims),
        fcf_y1=float(fcf_y1),
        n_years=int(n_years),
        g_inf=float(g_inf),
        min_spread=MIN_SPREAD,
        g_min=float(g_min), g_mode=float(g_mode), g_max=float(g_max),
        w_min=float(w_min), w_mode=float(w_mode), w_max=float(w_max),
        capex_min=float(capex_min), capex_mode=float(capex_mode), capex_max=float(capex_max),
    )

    if idx.size == 0:
        st.error("Configuración inconsistente: WACC no supera g∞ en escenarios suficientes. Ajuste parámetros.")
        st.stop()

    prob_neg = float(np.nanmean(npv_s < 0))
    p5, p50, p95 = np.nanpercentile(npv_s, [5, 50, 95])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("P(VAN<0)", f"{prob_neg:.1%}")
    m2.metric("P5", money(float(p5), currency))
    m3.metric("P50", money(float(p50), currency))
    m4.metric("P95", money(float(p95), currency))
    m5.metric("Simulaciones", f"{int(sims):,}")

    st.markdown("### Interpretación (registro no técnico, académico)")
    if prob_neg <= max_prob_negative and p50 > 0:
        st.success(
            f"El análisis probabilístico indica una expectativa favorable de creación de valor "
            f"(P50 = {money(float(p50), currency)}), con probabilidad de destrucción de valor "
            f"({prob_neg:.1%}) compatible con el umbral definido."
        )
    else:
        st.warning(
            f"El análisis probabilístico sugiere potencial de creación de valor (P50 = {money(float(p50), currency)}); "
            f"sin embargo, la probabilidad de destrucción de valor ({prob_neg:.1%}) resulta relevante "
            f"respecto del umbral definido. Se recomienda fortalecer supuestos, evidencia y mitigaciones."
        )

    fig = px.histogram(
        x=npv_s[~np.isnan(npv_s)],
        nbins=50,
        labels={"x": "VAN"},
        title="Distribución del VAN (Monte Carlo)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sensibilidad (señal)
    rng = np.random.default_rng()
    sample_n = min(2000, idx.size)
    sample_idx = rng.choice(idx, size=sample_n, replace=False)
    npv_sample = npv_s[sample_idx]

    corr_g = safe_corr(npv_sample, g_s[sample_idx])
    corr_w = safe_corr(npv_sample, w_s[sample_idx])
    corr_c = safe_corr(npv_sample, capex_s[sample_idx])

    drivers = []
    if not np.isnan(corr_g) and abs(corr_g) > 0.2:
        drivers.append("crecimiento (g)")
    if not np.isnan(corr_w) and abs(corr_w) > 0.2:
        drivers.append("tasa de descuento (WACC)")
    if not np.isnan(corr_c) and abs(corr_c) > 0.2:
        drivers.append("inversión (CAPEX)")
    driver_focus = ", ".join(drivers) if drivers else None

    if driver_focus:
        st.caption(f"Supuestos con sensibilidad relevante: **{driver_focus}** (señal orientativa; no implica causalidad).")
    else:
        st.caption("No se identificó una señal fuerte de sensibilidad (o la señal es limitada).")

    # Veredicto
    verdict = "SIN VEREDICTO"
    rationale = "Modo comité desactivado."
    criteria_lines = []

    if committee_mode:
        verdict, rationale = committee_verdict(
            prob_neg=float(prob_neg),
            p50=float(p50),
            p5=float(p5),
            max_prob_negative=float(max_prob_negative),
            require_p50_positive=bool(require_p50_positive),
            use_p5_floor=bool(use_p5_floor),
            p5_floor=float(p5_floor) if (use_p5_floor and p5_floor is not None) else None
        )

        criteria_lines.append(f"P(VAN<0) ≤ {max_prob_negative:.0%} → {'Cumple' if prob_neg <= max_prob_negative else 'No cumple'}")
        if require_p50_positive:
            criteria_lines.append(f"P50(VAN) > 0 → {'Cumple' if float(p50) > 0 else 'No cumple'}")
        if use_p5_floor and p5_floor is not None:
            criteria_lines.append(f"P5(VAN) ≥ {money(float(p5_floor), currency)} → {'Cumple' if float(p5) >= float(p5_floor) else 'No cumple'}")

        st.subheader("Veredicto del Comité")
        if verdict == "APROBADO":
            st.success(f"{verdict} {badge(verdict)} — {rationale}")
        elif verdict == "OBSERVADO":
            st.warning(f"{verdict} {badge(verdict)} — {rationale}")
        else:
            st.error(f"{verdict} {badge(verdict)} — {rationale}")

        st.markdown("**Criterios aplicados:**")
        st.markdown("\n".join([f"- {x}" for x in criteria_lines]))

    actions = recommended_actions(
        prob_neg=float(prob_neg),
        p5=float(p5),
        p50=float(p50),
        capex_base=float(capex_mode),
        threshold=float(max_prob_negative),
        driver_focus=driver_focus
    )
    st.subheader("Recomendación y plan de acción")
    for a in actions:
        st.write(f"• {a}")

    limitations = [
        "Los resultados dependen de supuestos de crecimiento, inversión y tasa; corresponde respaldarlos con evidencia empírica.",
        "El modelo de FCF es parsimonioso; en evaluación profesional se recomienda desagregar por drivers y reinversión/capital de trabajo.",
        "En Paraguay, valores de mercado de deuda/equity pueden no ser observables; el uso de valores contables o estimados es una aproximación defendible si se declara.",
        "El tratamiento del riesgo país (CRP) responde a un enfoque aplicado; se recomienda declarar el enfoque y su justificación.",
        "La TIR puede no ser única o no existir en flujos no convencionales; se prioriza VAN como criterio principal."
    ]

    crp_approach = (
        "Se incorpora como prima adicional en Ke (enfoque aplicado, frecuente en emergentes)."
        if use_crp else
        "No se incorpora explícitamente; se asume capturado por ERP y/o parámetros del modelo."
    )

    report = ExecReport(
        institution=institution.strip() or "Universidad San Ignacio de Loyola (USIL)",
        program=program.strip() or "MBA",
        course=course.strip() or "Curso/Módulo",
        project=project.strip() or "Proyecto",
        responsible=responsible.strip() or "Responsable",
        currency=currency,
        basis=basis,
        d_e_basis=d_e_basis,
        crp_approach=crp_approach,
        capex0=float(capex0),
        wacc=float(wacc),
        base_npv=float(base_npv),
        base_irr=base_irr,
        sims=int(sims),
        prob_neg=float(prob_neg),
        p5=float(p5),
        p50=float(p50),
        p95=float(p95),
        verdict=verdict,
        rationale=rationale,
        criteria_lines=criteria_lines,
        driver_focus=driver_focus,
        actions=actions,
        limitations=limitations
    )

    st.divider()
    st.header("🧾 Informe Executive (redacción automática en la app)")
    st.text_area("Informe (para copiar/pegar)", value=build_executive_text(report), height=420)

    st.subheader("📄 Exportación PDF (2 páginas)")
    if not REPORTLAB_OK:
        st.warning(
            "PDF deshabilitado: falta la librería `reportlab` en el entorno. "
            "Solución: agregá `reportlab` a tu `requirements.txt` y redeploy."
        )
    else:
        if st.button("Generar PDF institucional (2 páginas)"):
            try:
                pdf_bytes = generate_pdf_2pages(report, logo_reader=logo_reader)
                st.download_button(
                    "⬇️ Descargar PDF",
                    data=pdf_bytes,
                    file_name=f"USIL_MBA_Informe_Ejecutivo_{report.project.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"No se pudo generar el PDF: {e}")

else:
    st.info("Monte Carlo está desactivado. Activá el módulo para generar percentiles y el informe probabilístico.")
