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
def pct(x: float) -> str:
    return f"{x:.2%}"

def money(x: float, currency: str = "USD") -> str:
    # Formato simple y consistente. Si querés “Gs.”, lo ajustamos.
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
    # Filtro conservador: evita irr absurdas por problemas numéricos
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
            "Documentar supuestos clave (demanda, precio, margen, inversión, tasa) y su justificación, con indicadores observables."
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
            "Reducir incertidumbre en variables críticas mediante evidencia adicional y/o rediseño de supuestos, "
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
        "Líneas de acción: (i) faseo/racionalización del CAPEX, (ii) validación de demanda con indicadores, "
        "(iii) optimización de márgenes/costos, (iv) mitigación contractual y de ejecución."
    )

    if driver_focus:
        actions.append(
            f"Tratar como supuestos críticos aquellos vinculados a: {driver_focus}, priorizando evidencia y mitigación."
        )

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
):
    rng = np.random.default_rng()

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

    fcf_valid = fcf_paths[idx, :]
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

    # Resumen ejecutivo (3 bullets)
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
    lines.append(f"- WACC: {pct(r.wacc)} | Ke: {pct(r.ke)} | Kd: {pct(r.kd)}")
    lines.append(f"- Parámetros: Rf {pct(r.rf)} | ERP {pct(r.erp)} | CRP {pct(r.crp)} | βU {r.beta_u:.2f} | βL {r.beta_l:.2f}")
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
    irr_text = pct(report.base_irr) if report.base_irr is not None else "N/A (posible no unicidad/no existencia)"
    y = bullets(y, [
        f"CAPEX (Año 0): {money(report.capex0, report.currency)} ({report.currency})",
        f"WACC: {pct(report.wacc)} | Ke: {pct(report.ke)} | Kd: {pct(report.kd)}",
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
st.sidebar.header("2) Estructura de capital (para WACC
