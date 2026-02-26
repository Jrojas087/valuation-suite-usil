import streamlit as st
import numpy as np
import numpy_financial as npf
import plotly.express as px
from dataclasses import dataclass
from datetime import date
import io
import urllib.request

# ============================================================
# PDF (ReportLab) – import protegido
# ============================================================
REPORTLAB_OK = True
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
except Exception:
    REPORTLAB_OK = False

# Guardrails
MIN_SPREAD = 0.005  # WACC debe ser > g∞ + 0.5%
DEFAULT_MAX_CHARS = 105


# ============================================================
# UI base
# ============================================================
st.set_page_config(page_title="ValuationApp – MBA USIL (Paraguay)", layout="wide")
st.title("📊 ValuationApp – Evaluación de Proyectos (Comité Académico MBA – USIL)")
st.caption(
    "Marco: DCF + evaluación probabilística (Monte Carlo). "
    "Incluye explicación metodológica, criterios de comité e informe ejecutivo."
)

# ============================================================
# Helpers
# ============================================================
def pct_fmt(x: float) -> str:
    return f"{x:.2%}"

def money(x: float, currency: str = "USD") -> str:
    # Formato simple; si usás PYG, podés cambiar símbolo.
    if currency == "PYG":
        return f"Gs. {x:,.0f}".replace(",", ".")
    return f"${x:,.0f}"

def tri_ok(a, m, b) -> bool:
    return a <= m <= b

def safe_irr(x):
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    if np.isnan(x) or np.isinf(x):
        return None
    # Filtro conservador: evita IRR absurdas por problemas numéricos
    if x < -0.99 or x > 2.0:
        return None
    return x

def safe_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 50:
        return np.nan
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def badge(verdict: str) -> str:
    return {"APROBADO": "✅", "OBSERVADO": "⚠️", "RECHAZADO": "⛔"}.get(verdict, "—")

def _wrap_lines(text: str, max_chars: int = DEFAULT_MAX_CHARS):
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

def detect_non_conventional_flows(cashflows) -> bool:
    # Flujos no convencionales: más de un cambio de signo
    signs = []
    for x in cashflows:
        if abs(x) < 1e-12:
            continue
        signs.append(1 if x > 0 else -1)
    if len(signs) < 2:
        return False
    changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    return changes > 1


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
            "El perfil riesgo–retorno resulta incompatible con una decisión favorable en su estado actual."
        )
    return (
        "OBSERVADO",
        "El proyecto presenta potencial de creación de valor; sin embargo, puede presentar debilidades "
        "en supuestos críticos, justificando reforzar evidencia y mitigaciones antes de una aprobación."
    )

def recommended_actions(prob_neg, p5, p50, capex_base, threshold, driver_focus):
    actions = []

    if (prob_neg <= threshold) and (p50 > 0):
        actions.append(
            "Avanzar a la siguiente etapa, consolidando la trazabilidad de supuestos mediante evidencia empírica verificable "
            "(fuentes secundarias y levantamiento primario selectivo)."
        )
        actions.append(
            "Establecer hitos de control (control formativo) para validar supuestos críticos antes de comprometer capital adicional."
        )
        actions.append(
            "Documentar supuestos clave (demanda, precio, margen, inversión, tasa) y su justificación, con indicadores observables."
        )
        if driver_focus:
            actions.append(f"Priorizar evidencia y mitigaciones sobre determinantes con sensibilidad relevante: {driver_focus}.")
        return actions

    actions.append("Fortalecer supuestos fundamentales del caso antes de una aprobación, dado el perfil de riesgo evidenciado.")

    if prob_neg > threshold:
        actions.append(
            "Reducir incertidumbre en variables críticas mediante evidencia adicional y/o rediseño de supuestos, "
            "dado que P(VAN<0) resulta significativa respecto del umbral definido."
        )
    if p50 <= 0:
        actions.append("Revisar modelo de ingresos/costos y estructura de inversión para robustecer el caso base (P50 no favorable).")

    if p5 < -0.15 * capex_base:
        actions.append(
            "Considerar una estrategia por fases (piloto → escalamiento) para reducir exposición inicial, "
            "dado un downside material en P5."
        )

    actions.append(
        "Líneas de acción sugeridas: (i) faseo/racionalización del CAPEX, (ii) validación de demanda con indicadores, "
        "(iii) optimización de márgenes/costos, (iv) mitigación contractual y de ejecución."
    )

    if driver_focus:
        actions.append(f"Tratar como supuestos críticos aquellos vinculados a: {driver_focus}, priorizando evidencia y mitigación.")

    return actions


# ============================================================
# Monte Carlo (incluye shock al nivel de FCF Año 1)
# ============================================================
@st.cache_data(show_spinner=False)
def run_monte_carlo(
    sims: int,
    fcf_y1: float,
    n_years: int,
    g_inf: float,
    min_spread: float,
    # crecimiento g (triangular)
    g_min: float, g_mode: float, g_max: float,
    # WACC (triangular)
    w_min: float, w_mode: float, w_max: float,
    # CAPEX (triangular)
    capex_min: float, capex_mode: float, capex_max: float,
    # Shock FCF nivel (triangular)
    fcf_mult_min: float, fcf_mult_mode: float, fcf_mult_max: float,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    g_s = rng.triangular(g_min, g_mode, g_max, sims)
    w_s = rng.triangular(w_min, w_mode, w_max, sims)
    capex_s = rng.triangular(capex_min, capex_mode, capex_max, sims)
    mult_s = rng.triangular(fcf_mult_min, fcf_mult_mode, fcf_mult_max, sims)

    yrs = np.arange(1, n_years + 1)
    # FCF Año 1 impactado por multiplicador (shock de demanda/margen)
    fcf1_s = fcf_y1 * mult_s
    fcf_paths = fcf1_s[:, None] * (1.0 + g_s)[:, None] ** (yrs[None, :] - 1)

    # Consistencia TV
    valid = w_s > (g_inf + min_spread)

    npv_s = np.full(sims, np.nan)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return npv_s, g_s, w_s, capex_s, mult_s, idx

    fcf_valid = fcf_paths[idx, :].copy()
    w_valid = w_s[idx]
    capex_valid = capex_s[idx]

    tv_valid = (fcf_valid[:, -1] * (1.0 + g_inf)) / (w_valid - g_inf)
    fcf_valid[:, -1] += tv_valid

    discount = (1.0 + w_valid)[:, None] ** (yrs[None, :])
    pv = np.sum(fcf_valid / discount, axis=1)

    npv_s[idx] = pv - capex_valid
    return npv_s, g_s, w_s, capex_s, mult_s, idx


# ============================================================
# Informe ejecutivo
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
    ke: float
    kd: float
    rf: float
    erp: float
    crp: float
    beta_u: float
    beta_l: float

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
    irr_text = pct_fmt(r.base_irr) if r.base_irr is not None else "N/A (posible no unicidad/no existencia)"

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

    lines.append("RESUMEN EJECUTIVO")
    lines.append(f"- Dictamen: {r.verdict} {badge(r.verdict)}")
    lines.append(f"- Riesgo: P(VAN<0) = {r.prob_neg:.1%} | P5 = {money(r.p5, r.currency)} | P50 = {money(r.p50, r.currency)}")
    lines.append("- Próximo paso: validar supuestos críticos con evidencia y formalizar mitigaciones antes de comprometer capital incremental.")
    lines.append("")

    lines.append(f"1. Síntesis ejecutiva – Veredicto: {r.verdict} {badge(r.verdict)}")
    lines.append(r.rationale)
    lines.append("")
    lines.append("2. Indicadores principales")
    lines.append(f"- CAPEX (Año 0): {money(r.capex0, r.currency)} ({r.currency})")
    lines.append(f"- WACC: {pct_fmt(r.wacc)} | Ke: {pct_fmt(r.ke)} | Kd: {pct_fmt(r.kd)}")
    lines.append(f"- Parámetros: Rf {pct_fmt(r.rf)} | ERP {pct_fmt(r.erp)} | CRP {pct_fmt(r.crp)} | βU {r.beta_u:.2f} | βL {r.beta_l:.2f}")
    lines.append(f"- VAN base (determinístico): {money(r.base_npv, r.currency)}")
    lines.append(f"- TIR base (determinística): {irr_text}")
    lines.append(f"- Monte Carlo: {r.sims:,} simulaciones")
    lines.append(f"- P(VAN<0): {r.prob_neg:.1%}")
    lines.append(f"- P5: {money(r.p5, r.currency)} | P50: {money(r.p50, r.currency)} | P95: {money(r.p95, r.currency)}")
    lines.append("")
    lines.append("3. Lectura probabilística del riesgo (registro no técnico)")
    lines.append(
        f"El análisis Monte Carlo caracteriza la distribución del VAN bajo incertidumbre razonable. "
        f"El resultado central (P50) se estima en {money(r.p50, r.currency)}; el escenario adverso plausible (P5) "
        f"alcanza {money(r.p5, r.currency)}. La probabilidad estimada de destrucción de valor P(VAN<0) es {r.prob_neg:.1%}."
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
    lines.append("5. Supuestos y consistencia metodológica (Paraguay)")
    lines.append(f"- Moneda: {r.currency}")
    lines.append(f"- Base de medición (tasa y flujos): {r.basis}")
    lines.append(f"- Estructura D/E utilizada para WACC: {r.d_e_basis}")
    lines.append(f"- Riesgo país (CRP): {r.crp_approach}")
    lines.append(
        "- Nota de contexto: dada la limitada profundidad del mercado de capitales local, valores de mercado de deuda/equity "
        "pueden no ser observables; se admite el uso de valores contables o estimados, siempre que se declare y justifique."
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


# ============================================================
# PDF
# ============================================================
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

def generate_pdf_2pages(report: ExecReport, logo_reader=None) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab no está disponible. Agregue `reportlab` a requirements.txt.")

    tmp = io.BytesIO()
    c = canvas.Canvas(tmp, pagesize=letter)
    width, height = letter

    left = 0.75 * inch
    top = height - 0.75 * inch
    leading = 12

    def header(y, page_no):
        if logo_reader is not None:
            c.drawImage(logo_reader, left, y - 40, width=120, height=40, mask="auto")
            c.setFont("Helvetica-Bold", 11.5)
            c.drawString(left + 130, y - 12, report.institution)
            c.setFont("Helvetica", 10.5)
            c.drawString(left + 130, y - 28, report.program)
            c.drawString(left + 130, y - 42, report.course)
        else:
            c.setFont("Helvetica-Bold", 11.5)
            c.drawString(left, y, report.institution)
            c.setFont("Helvetica", 10.5)
            c.drawString(left, y - 14, report.program)
            c.drawString(left, y - 28, report.course)

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
        for line in _wrap_lines(text, DEFAULT_MAX_CHARS):
            c.drawString(left, y, line)
            y -= leading
        return y

    def bullets(y, items):
        c.setFont("Helvetica", 10.5)
        for it in items:
            for line in _wrap_lines(f"• {it}", DEFAULT_MAX_CHARS):
                c.drawString(left, y, line)
                y -= leading
        return y

    # Page 1
    y = header(top, 1)
    y = h1(y, "Informe Ejecutivo – Evaluación de Proyecto (2 páginas)")
    y = p(y, f"Proyecto: {report.project}  |  Responsable: {report.responsible}")
    y -= 6

    y = h2(y, f"Resumen ejecutivo – Dictamen: {report.verdict} {badge(report.verdict)}")
    y = bullets(y, [
        f"Riesgo: P(VAN<0) = {report.prob_neg:.1%} | P5 = {money(report.p5, report.currency)} | P50 = {money(report.p50, report.currency)}",
        "Próximo paso: validar supuestos críticos con evidencia y formalizar mitigaciones antes de comprometer capital incremental."
    ])
    y -= 6

    y = h2(y, "Indicadores principales")
    irr_text = pct_fmt(report.base_irr) if report.base_irr is not None else "N/A (posible no unicidad/no existencia)"
    y = bullets(y, [
        f"CAPEX (Año 0): {money(report.capex0, report.currency)} ({report.currency})",
        f"WACC: {pct_fmt(report.wacc)} | Ke: {pct_fmt(report.ke)} | Kd: {pct_fmt(report.kd)}",
        f"VAN base: {money(report.base_npv, report.currency)} | TIR base: {irr_text}",
        f"Monte Carlo: {report.sims:,} simulaciones | P(VAN<0) {report.prob_neg:.1%}",
        f"P5: {money(report.p5, report.currency)} | P50: {money(report.p50, report.currency)} | P95: {money(report.p95, report.currency)}",
    ])
    y -= 6

    y = h2(y, "Lectura probabilística (no técnica)")
    y = p(y,
          f"El resultado central (P50) se estima en {money(report.p50, report.currency)}; "
          f"el escenario adverso plausible (P5) en {money(report.p5, report.currency)}. "
          f"La probabilidad de VAN negativo es {report.prob_neg:.1%}."
          )
    if report.driver_focus:
        y = p(y, f"Sensibilidad relevante asociada a: {report.driver_focus} (señal orientativa).")

    y -= 4
    y = h2(y, "Criterios del Comité")
    y = bullets(y, report.criteria_lines if report.criteria_lines else ["Modo comité desactivado."])

    c.showPage()

    # Page 2
    y = header(top, 2)
    y = h1(y, "Informe Ejecutivo – Continuación")

    y = h2(y, "Supuestos y consistencia metodológica (Paraguay)")
    y = bullets(y, [
        f"Moneda: {report.currency}.",
        f"Base de medición (tasa y flujos): {report.basis}.",
        f"Estructura D/E para WACC: {report.d_e_basis}.",
        f"Riesgo país (CRP): {report.crp_approach}.",
        "Nota: valores de mercado de deuda/equity pueden no ser observables en Paraguay; se admite valores contables o estimados si se declara y justifica.",
    ])
    y -= 6

    y = h2(y, "Recomendación y plan de acción")
    y = bullets(y, report.actions[:8])
    y -= 6

    y = h2(y, "Limitaciones (declaración académica)")
    y = bullets(y, report.limitations[:8])

    c.save()
    return tmp.getvalue()


# ============================================================
# Sidebar – Identidad institucional + contexto Paraguay
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
st.sidebar.header("🇵🇾 Contexto Paraguay")
currency = st.sidebar.selectbox("Moneda del modelo", ["USD", "PYG"], index=0)
basis = st.sidebar.selectbox("Base de medición (tasa y flujos)", ["Nominal", "Real"], index=0)
d_e_basis = st.sidebar.selectbox(
    "Base D/E para WACC",
    ["Valores contables (práctica frecuente en PY)", "Mixto/estimado (si hay evidencia)", "Valores de mercado (si disponible)"],
    index=0
)

st.sidebar.divider()
st.sidebar.header("🖼️ Logo para PDF (opcional)")
logo_file = st.sidebar.file_uploader("Subir logo (PNG/JPG)", type=["png", "jpg", "jpeg"])
logo_url = st.sidebar.text_input("o URL directa de imagen (.png/.jpg)", value="").strip()
logo_reader = _load_logo_reader(logo_file, logo_url if logo_url else None)

st.sidebar.caption("Para PDF institucional: ideal subir un PNG/JPG al repositorio o usar este uploader.")


# ============================================================
# Inputs financieros
# ============================================================
st.sidebar.divider()
st.sidebar.header("0) Inversión inicial")
capex0 = st.sidebar.number_input("CAPEX Año 0", value=500000.0, step=10000.0, min_value=1.0)

st.sidebar.divider()
st.sidebar.header("1) CAPM / WACC")
rf = st.sidebar.number_input("Tasa libre de riesgo Rf (%)", value=4.5, step=0.1) / 100
erp = st.sidebar.number_input("Prima de riesgo de mercado (ERP %) ", value=5.5, step=0.1) / 100

use_crp = st.sidebar.checkbox("Incluir Riesgo País (CRP) en Ke", value=True)
crp = st.sidebar.number_input("CRP (%)", value=2.0, step=0.1) / 100 if use_crp else 0.0

beta_u = st.sidebar.number_input("βU (desapalancada)", value=0.90, step=0.05)
tax_rate = st.sidebar.number_input("Impuesto T (%)", value=10.0, step=0.5) / 100

st.sidebar.divider()
st.sidebar.header("2) Estructura de capital (para WACC)")
debt = st.sidebar.number_input("Deuda D", value=400000.0, step=10000.0, min_value=0.0)
equity = st.sidebar.number_input("Capital propio E", value=600000.0, step=10000.0, min_value=1.0)
kd = st.sidebar.number_input("Costo de deuda Kd (%)", value=7.0, step=0.1) / 100

st.sidebar.divider()
st.sidebar.header("3) Proyecciones (DCF)")
n_years = st.sidebar.slider("Años de proyección explícita", 1, 10, 5)
fcf_y1 = st.sidebar.number_input("FCF Año 1", value=100000.0, step=5000.0, min_value=0.0)
g_exp = st.sidebar.number_input("g (crecimiento explícito %) ", value=5.0, step=0.25) / 100
g_inf = st.sidebar.number_input("g∞ (crecimiento perpetuo %) ", value=2.0, step=0.10) / 100

st.sidebar.caption("Regla técnica: para TV (Gordon), se requiere WACC > g∞. En Monte Carlo se exige WACC > g∞ + 0.5%.")

st.sidebar.divider()
st.sidebar.header("4) Monte Carlo")
use_mc = st.sidebar.checkbox("Activar simulación Monte Carlo", value=True)
sims = st.sidebar.slider("Simulaciones", 1000, 50000, 10000, 1000)
seed = st.sidebar.number_input("Semilla (reproducibilidad)", value=42, step=1)

st.sidebar.subheader("Rangos triangulares – g (explícito)")
g_min = st.sidebar.number_input("g mínimo (%)", value=1.0, step=0.25) / 100
g_mode = st.sidebar.number_input("g más probable (%)", value=5.0, step=0.25) / 100
g_max = st.sidebar.number_input("g máximo (%)", value=9.0, step=0.25) / 100

st.sidebar.subheader("Rangos triangulares – CAPEX (Año 0)")
capex_min = st.sidebar.number_input("CAPEX mínimo", value=450000.0, step=10000.0, min_value=1.0)
capex_mode = st.sidebar.number_input("CAPEX más probable", value=500000.0, step=10000.0, min_value=1.0)
capex_max = st.sidebar.number_input("CAPEX máximo", value=600000.0, step=10000.0, min_value=1.0)

st.sidebar.subheader("Shock al nivel de FCF Año 1 (multiplicador)")
st.sidebar.caption("Representa shocks de demanda/margen. Ej.: 0.9 = -10%, 1.1 = +10%.")
fcf_mult_min = st.sidebar.number_input("Multiplicador mínimo", value=0.85, step=0.01, min_value=0.01)
fcf_mult_mode = st.sidebar.number_input("Multiplicador más probable", value=1.00, step=0.01, min_value=0.01)
fcf_mult_max = st.sidebar.number_input("Multiplicador máximo", value=1.15, step=0.01, min_value=0.01)

st.sidebar.subheader("Rangos triangulares – WACC")
auto_wacc = st.sidebar.checkbox("Auto WACC (±2% absoluto alrededor del WACC calculado)", value=True)
if auto_wacc:
    w_min = None
    w_mode = None
    w_max = None
else:
    w_min = st.sidebar.number_input("WACC mínimo (%)", value=9.0, step=0.25) / 100
    w_mode = st.sidebar.number_input("WACC más probable (%)", value=11.0, step=0.25) / 100
    w_max = st.sidebar.number_input("WACC máximo (%)", value=13.0, step=0.25) / 100

st.sidebar.divider()
st.sidebar.header("5) Criterios del Comité")
committee_on = st.sidebar.checkbox("Activar dictamen automático (comité)", value=True)
max_prob_negative = st.sidebar.slider("Umbral máximo P(VAN<0)", 0.0, 0.8, 0.2, 0.01)
require_p50_positive = st.sidebar.checkbox("Exigir P50(VAN) > 0", value=True)
use_p5_floor = st.sidebar.checkbox("Exigir P5(VAN) ≥ piso", value=False)
p5_floor = st.sidebar.number_input("Piso P5 (si aplica)", value=0.0, step=10000.0)

st.sidebar.divider()
st.sidebar.header("6) Limitaciones (para el informe)")
lim_defaults = [
    "Los resultados dependen de la calidad de los supuestos; la herramienta no reemplaza evidencia empírica.",
    "La estimación de WACC y CRP puede variar según fuentes y metodología; se recomienda documentar racionalidad.",
    "El valor terminal (Gordon) requiere consistencia macro y WACC > g∞; su sensibilidad debe ser evaluada.",
    "La simulación Monte Carlo refleja incertidumbre según rangos definidos por el usuario; no predice el futuro."
]
limitations = st.sidebar.multiselect("Seleccionar limitaciones", lim_defaults, default=lim_defaults)


# ============================================================
# Cálculos determinísticos (DCF + WACC)
# ============================================================
beta_l = beta_u * (1.0 + (1.0 - tax_rate) * (debt / equity))
ke = rf + beta_l * erp + (crp if use_crp else 0.0)

total_cap = debt + equity
wacc = (equity / total_cap) * ke + (debt / total_cap) * kd * (1.0 - tax_rate)

# DCF flujos
years = np.arange(1, n_years + 1)
fcf = np.array([fcf_y1 * (1.0 + g_exp) ** (t - 1) for t in years], dtype=float)

# Validación TV
if wacc <= g_inf:
    st.error("Condición inválida: WACC debe ser mayor que g∞ para calcular el valor terminal (Gordon-Shapiro).")
    st.stop()

tv = (fcf[-1] * (1.0 + g_inf)) / (wacc - g_inf)

# VAN determinístico
disc = (1.0 + wacc) ** years
pv = np.sum(fcf / disc) + tv / ((1.0 + wacc) ** n_years)
base_npv = pv - capex0

# TIR determinística
cash_for_irr = np.concatenate(([-capex0], fcf[:-1], [fcf[-1] + tv]))
irr_raw = npf.irr(cash_for_irr)
base_irr = safe_irr(irr_raw)

nonconv = detect_non_conventional_flows(cash_for_irr)

# ============================================================
# Vista principal – métricas
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("βU / βL", f"{beta_u:.2f} / {beta_l:.2f}")
c2.metric("Ke / Kd", f"{pct_fmt(ke)} / {pct_fmt(kd)}")
c3.metric("WACC", pct_fmt(wacc))
c4.metric("VAN (base)", money(base_npv, currency))

st.caption("Nota: si hay flujos no convencionales, la TIR puede no ser única o no ser interpretable como criterio primario.")
if nonconv:
    st.warning("Se detectaron flujos **no convencionales** (más de un cambio de signo). En comité, el criterio preferente es el **VAN**.")

st.write("**TIR (base):** " + (pct_fmt(base_irr) if base_irr is not None else "N/A (posible no unicidad/no existencia)"))

# ============================================================
# Gráfico de flujos
# ============================================================
st.subheader("Proyección de flujos de caja (período explícito)")
df_flows = {"Año": years, "FCF": fcf}
fig_flows = px.bar(df_flows, x="Año", y="FCF", title="FCF (Explícito)")
st.plotly_chart(fig_flows, use_container_width=True)

st.info(f"**Valor Terminal (TV)** incorporado al final del año {n_years}: {money(tv, currency)}")

# ============================================================
# Explicaciones (in-app)
# ============================================================
with st.expander("📚 Explicaciones (g vs g∞, DCF, βU/βL, percentiles y flujos no convencionales)"):
    st.markdown(
        f"""
**g (crecimiento explícito) vs g∞ (crecimiento perpetuo)**  
- **g (explícito)**: tasa aplicada durante los **años proyectados** (1..N).  
- **g∞**: tasa de crecimiento de largo plazo utilizada para estimar el **valor terminal** (continuidad del negocio).  
Regla técnica: para un valor terminal coherente se requiere **WACC > g∞**.

**DCF (Discounted Cash Flow)**  
El DCF estima el valor económico descontando los flujos futuros (FCF) a una tasa que refleje el riesgo (**WACC**).  
En esta aplicación: valor presente de FCF (1..N) + valor terminal descontado – CAPEX (año 0).

**βU y βL (Hamada)**  
- **βU** (desapalancada): riesgo del negocio sin efecto de deuda.  
- **βL** (apalancada): riesgo del equity considerando deuda (D/E). A mayor apalancamiento, mayor βL y, típicamente, mayor Ke.

**P5, P50, P95 (Monte Carlo)**  
Percentiles del VAN simulado:  
- **P50**: mediana (resultado “central” probabilístico).  
- **P5**: escenario adverso plausible.  
- **P95**: escenario favorable plausible.

**Flujos no convencionales**  
Si los flujos cambian de signo más de una vez, la **TIR** puede ser múltiple o no representar adecuadamente la decisión.  
En comité, se privilegia el **VAN** (y el análisis de riesgo).
"""
    )

# ============================================================
# Monte Carlo + comité
# ============================================================
st.header("🎲 Monte Carlo y Comité")
if use_mc:
    if not tri_ok(g_min, g_mode, g_max):
        st.error("Rango triangular inválido para g: debe cumplirse g_min ≤ g_mode ≤ g_max.")
        st.stop()
    if not tri_ok(capex_min, capex_mode, capex_max):
        st.error("Rango triangular inválido para CAPEX: debe cumplirse min ≤ mode ≤ max.")
        st.stop()
    if not tri_ok(fcf_mult_min, fcf_mult_mode, fcf_mult_max):
        st.error("Rango triangular inválido para multiplicador FCF: min ≤ mode ≤ max.")
        st.stop()

    if auto_wacc:
        w_min_mc = max(0.0001, wacc - 0.02)
        w_mode_mc = max(0.0001, wacc)
        w_max_mc = max(0.0001, wacc + 0.02)
    else:
        if not tri_ok(w_min, w_mode, w_max):
            st.error("Rango triangular inválido para WACC: debe cumplirse min ≤ mode ≤ max.")
            st.stop()
        w_min_mc, w_mode_mc, w_max_mc = w_min, w_mode, w_max

    # Asegurar spread mínimo para TV
    if w_min_mc <= g_inf + MIN_SPREAD:
        st.warning("WACC mínimo es demasiado cercano a g∞. Se ajustará internamente para mantener coherencia en TV.")
        w_min_mc = g_inf + MIN_SPREAD + 0.0005
        w_mode_mc = max(w_mode_mc, g_inf + MIN_SPREAD + 0.0010)
        w_max_mc = max(w_max_mc, g_inf + MIN_SPREAD + 0.0015)

    npv_s, g_s, w_s, capex_s, mult_s, idx = run_monte_carlo(
        sims=sims,
        fcf_y1=fcf_y1,
        n_years=n_years,
        g_inf=g_inf,
        min_spread=MIN_SPREAD,
        g_min=g_min, g_mode=g_mode, g_max=g_max,
        w_min=w_min_mc, w_mode=w_mode_mc, w_max=w_max_mc,
        capex_min=capex_min, capex_mode=capex_mode, capex_max=capex_max,
        fcf_mult_min=fcf_mult_min, fcf_mult_mode=fcf_mult_mode, fcf_mult_max=fcf_mult_max,
        seed=int(seed),
    )

    valid_npvs = npv_s[~np.isnan(npv_s)]
    if valid_npvs.size < 100:
        st.error("Muy pocas simulaciones válidas. Revise rangos de WACC y g∞ (condición WACC > g∞ + spread).")
        st.stop()

    p5 = float(np.percentile(valid_npvs, 5))
    p50 = float(np.percentile(valid_npvs, 50))
    p95 = float(np.percentile(valid_npvs, 95))
    prob_neg = float(np.mean(valid_npvs < 0))

    # Correlaciones (orientativas)
    corr_g = safe_corr(valid_npvs, g_s[~np.isnan(npv_s)])
    corr_w = safe_corr(valid_npvs, w_s[~np.isnan(npv_s)])
    corr_capex = safe_corr(valid_npvs, capex_s[~np.isnan(npv_s)])
    corr_mult = safe_corr(valid_npvs, mult_s[~np.isnan(npv_s)])

    corrs = {
        "g (crecimiento explícito)": corr_g,
        "WACC": corr_w,
        "CAPEX": corr_capex,
        "Shock FCF (multiplicador)": corr_mult,
    }
    # driver_focus: el de mayor |corr|
    driver_focus = None
    if all(np.isnan(v) for v in corrs.values()) is False:
        driver_focus = max(corrs.items(), key=lambda kv: (0 if np.isnan(kv[1]) else abs(kv[1])))[0]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("P(VAN<0)", f"{prob_neg:.1%}")
    d2.metric("P5 (VAN)", money(p5, currency))
    d3.metric("P50 (VAN)", money(p50, currency))
    d4.metric("P95 (VAN)", money(p95, currency))

    st.subheader("Distribución del VAN (Monte Carlo)")
    fig_hist = px.histogram(x=valid_npvs, nbins=40, title="Distribución del VAN simulado")
    fig_hist.update_layout(xaxis_title="VAN", yaxis_title="Frecuencia")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Interpretación no técnica (lectura ejecutiva)")
    st.success(
        f"En términos probabilísticos, el resultado central (P50) se ubica en {money(p50, currency)}. "
        f"El escenario adverso plausible (P5) alcanza {money(p5, currency)}. "
        f"La probabilidad estimada de destrucción de valor P(VAN<0) es {prob_neg:.1%}."
    )

    if driver_focus:
        st.caption(f"Señal orientativa de sensibilidad: el VAN se asocia con variaciones en **{driver_focus}** (correlación como guía, no causalidad).")

    # Comité
    criteria_lines = []
    if committee_on:
        criteria_lines.append(f"P(VAN<0) ≤ {max_prob_negative:.0%}: {'Cumple' if prob_neg <= max_prob_negative else 'No cumple'}")
        if require_p50_positive:
            criteria_lines.append(f"P50(VAN) > 0: {'Cumple' if p50 > 0 else 'No cumple'}")
        if use_p5_floor:
            criteria_lines.append(f"P5(VAN) ≥ {money(p5_floor, currency)}: {'Cumple' if p5 >= p5_floor else 'No cumple'}")

        verdict, rationale = committee_verdict(
            prob_neg=prob_neg,
            p50=p50,
            p5=p5,
            max_prob_negative=max_prob_negative,
            require_p50_positive=require_p50_positive,
            use_p5_floor=use_p5_floor,
            p5_floor=p5_floor
        )
        st.subheader(f"Dictamen del Comité: {verdict} {badge(verdict)}")
        if verdict == "APROBADO":
            st.success(rationale)
        elif verdict == "OBSERVADO":
            st.warning(rationale)
        else:
            st.error(rationale)

        actions = recommended_actions(
            prob_neg=prob_neg,
            p5=p5,
            p50=p50,
            capex_base=capex0,
            threshold=max_prob_negative,
            driver_focus=driver_focus
        )

        st.markdown("**Recomendación y plan de acción (control formativo):**")
        for a in actions:
            st.write(f"- {a}")
    else:
        verdict, rationale = "N/A", "Modo comité desactivado."
        actions = []
else:
    st.info("Monte Carlo desactivado. Puede activarlo para obtener percentiles (P5/P50/P95) y P(VAN<0).")
    # valores mínimos para reporte
    prob_neg, p5, p50, p95 = np.nan, np.nan, np.nan, np.nan
    verdict, rationale = "N/A", "Simulación desactivada."
    criteria_lines, driver_focus, actions = [], None, []

# ============================================================
# Informe ejecutivo + PDF
# ============================================================
st.header("🧾 Informe Ejecutivo")
crp_approach = "CRP incorporado en Ke (CAPM extendido)" if use_crp else "CRP no incorporado (Ke sin prima país explícita)"

report = ExecReport(
    institution=institution,
    program=program,
    course=course,
    project=project,
    responsible=responsible,
    currency=currency,
    basis=basis,
    d_e_basis=d_e_basis,
    crp_approach=crp_approach,
    capex0=capex0,
    wacc=wacc,
    ke=ke,
    kd=kd,
    rf=rf,
    erp=erp,
    crp=crp if use_crp else 0.0,
    beta_u=beta_u,
    beta_l=beta_l,
    base_npv=base_npv,
    base_irr=base_irr,
    sims=int(sims) if use_mc else 0,
    prob_neg=float(prob_neg) if use_mc else 0.0,
    p5=float(p5) if use_mc else 0.0,
    p50=float(p50) if use_mc else 0.0,
    p95=float(p95) if use_mc else 0.0,
    verdict=verdict if committee_on else "N/A",
    rationale=rationale,
    criteria_lines=criteria_lines,
    driver_focus=driver_focus,
    actions=actions if actions else ["Definir supuestos, documentar evidencia y completar el modelo financiero antes de emitir dictamen."],
    limitations=limitations,
)

exec_text = build_executive_text(report)
st.text_area("Informe (copiar/pegar)", value=exec_text, height=360)

col_pdf1, col_pdf2 = st.columns([1, 2])
with col_pdf1:
    if REPORTLAB_OK:
        try:
            pdf_bytes = generate_pdf_2pages(report, logo_reader=logo_reader)
            st.download_button(
                "📥 Descargar PDF (2 páginas)",
                data=pdf_bytes,
                file_name=f"Informe_Ejecutivo_{project.replace(' ', '_')}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"No se pudo generar PDF: {e}")
    else:
        st.warning("PDF deshabilitado: falta `reportlab` en requirements.txt.")
with col_pdf2:
    st.caption("Sugerencia operativa: en Streamlit Cloud, agregá `reportlab` a `requirements.txt` para habilitar el PDF.")

st.divider()
st.caption("Fin del reporte. Esta herramienta tiene fines académicos y no sustituye evidencia empírica ni due diligence.")
