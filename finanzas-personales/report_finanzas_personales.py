# report_finanzas_personales.py
# Exporter (TXT + PDF) para la Consultoría de Finanzas Personales — sin matplotlib, PDF 100% ReportLab

from __future__ import annotations

import io
import math
import textwrap
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

REPORTLAB_OK = True
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
except Exception:
    REPORTLAB_OK = False


# ----------------------------
# Formatos
# ----------------------------
def fmt_pct(x: Optional[float]) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    return f"{x*100:.1f}%"

def fmt_pyg(x: Optional[float]) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    return "Gs. {:,.0f}".format(float(x)).replace(",", ".")

def wrap(s: str, width: int) -> list[str]:
    return textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False)


# ----------------------------
# Data model
# ----------------------------
@dataclass
class Rating:
    key: str      # "good" | "warn" | "bad"
    label: str
    emoji: str

@dataclass
class ClientFinReport:
    consultant: str
    client: str
    report_date: str
    age: Optional[int]
    occupation: str
    dependents: int
    objective: str

    # Perfil de riesgo
    risk_score: int
    risk_max: int
    risk_category: str
    risk_description: str
    risk_alloc_fixed: int   # % ilustrativo en instrumentos de bajo riesgo
    risk_alloc_variable: int  # % ilustrativo en instrumentos de mayor riesgo

    # Diagnóstico financiero (montos)
    income: float
    fixed_expenses: float
    variable_expenses: float
    debt_payment: float
    savings_monthly: float
    emergency_fund: float
    net_worth: Optional[float]

    # Indicadores calculados
    total_expenses: float
    cashflow: float
    savings_rate: float
    dti: float
    emergency_months: float

    cashflow_rating: Rating
    savings_rating: Rating
    dti_rating: Rating
    emergency_rating: Rating

    health_score: int
    health_max: int
    health_label: str

    action_plan: Sequence[str] = field(default_factory=list)

DISCLAIMER = (
    "Herramienta educativa del Diplomado de Finanzas Personales. No constituye asesoría de inversión "
    "regulada; valida cualquier decisión relevante con un asesor financiero certificado."
)


# ----------------------------
# TXT (para descarga rápida)
# ----------------------------
def build_text_report(r: ClientFinReport) -> str:
    lines: list[str] = []
    lines.append("DIAGNÓSTICO DE FINANZAS PERSONALES — REPORTE DE CONSULTORÍA")
    lines.append(f"Consultor/a: {r.consultant}  |  Cliente: {r.client}")
    lines.append(f"Fecha: {r.report_date}")
    edad = "—" if r.age is None else str(r.age)
    lines.append(f"Edad: {edad}  |  Ocupación: {r.occupation}  |  Dependientes: {r.dependents}")
    lines.append(f"Objetivo principal: {r.objective}")
    lines.append("")
    lines.append(f"PERFIL DE RIESGO: {r.risk_category}  ({r.risk_score}/{r.risk_max} pts)")
    lines.append(r.risk_description)
    lines.append(
        f"Mezcla ilustrativa (no es recomendación de inversión): "
        f"{r.risk_alloc_fixed}% bajo riesgo/ahorro — {r.risk_alloc_variable}% mayor riesgo/crecimiento"
    )
    lines.append("")
    lines.append("DIAGNÓSTICO FINANCIERO RÁPIDO")
    lines.append(f"- Ingreso mensual: {fmt_pyg(r.income)}")
    lines.append(f"- Gastos fijos: {fmt_pyg(r.fixed_expenses)} | Gastos variables: {fmt_pyg(r.variable_expenses)}")
    lines.append(f"- Cuota de deudas: {fmt_pyg(r.debt_payment)}")
    lines.append(f"- Ahorro/inversión mensual: {fmt_pyg(r.savings_monthly)}")
    lines.append(f"- Fondo de emergencia actual: {fmt_pyg(r.emergency_fund)}")
    if r.net_worth is not None:
        lines.append(f"- Patrimonio neto aproximado: {fmt_pyg(r.net_worth)}")
    lines.append("")
    lines.append("Indicadores")
    lines.append(f"- Flujo de caja mensual: {fmt_pyg(r.cashflow)}  [{r.cashflow_rating.emoji} {r.cashflow_rating.label}]")
    lines.append(f"- Tasa de ahorro: {fmt_pct(r.savings_rate)}  [{r.savings_rating.emoji} {r.savings_rating.label}]")
    lines.append(f"- Endeudamiento (cuota/ingreso): {fmt_pct(r.dti)}  [{r.dti_rating.emoji} {r.dti_rating.label}]")
    lines.append(f"- Fondo de emergencia: {r.emergency_months:.1f} meses cubiertos  [{r.emergency_rating.emoji} {r.emergency_rating.label}]")
    lines.append("")
    lines.append(f"SALUD FINANCIERA GENERAL: {r.health_label} ({r.health_score}/{r.health_max})")
    lines.append("")
    lines.append("PLAN DE ACCIÓN SUGERIDO")
    for item in r.action_plan:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


# ----------------------------
# PDF (ReportLab, estilo dark dashboard)
# ----------------------------
def generate_pdf(r: ClientFinReport) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab no está disponible. Agregar `reportlab` a requirements.txt.")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter

    bg = colors.HexColor("#050914")
    card = colors.HexColor("#0b1733")
    card2 = colors.HexColor("#0d1b3d")
    line = colors.Color(1, 1, 1, alpha=0.10)
    text = colors.HexColor("#EAF1FF")
    muted = colors.Color(234/255, 241/255, 1, alpha=0.75)
    accent = colors.HexColor("#66A9FF")
    good = colors.HexColor("#27D17C")
    warn = colors.HexColor("#FFCC66")
    bad = colors.HexColor("#FF5D5D")

    rating_color = {"good": good, "warn": warn, "bad": bad}

    def rr(x, y, w, h, r=14, fill=card):
        c.setFillColor(fill)
        c.setStrokeColor(line)
        c.setLineWidth(1)
        c.roundRect(x, y, w, h, r, stroke=1, fill=1)

    def t(x, y, s, size=10.5, bold=False, col=text):
        c.setFillColor(col)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x, y, s)

    def tr(x, y, s, size=10.5, bold=False, col=text):
        c.setFillColor(col)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawRightString(x, y, s)

    def para(x, y, s, width_chars=90, size=9.4, leading=12, col=muted):
        yy = y
        for ln in wrap(s, width_chars):
            t(x, yy, ln, size=size, bold=False, col=col)
            yy -= leading
        return yy

    # Background
    c.setFillColor(bg)
    c.rect(0, 0, W, H, stroke=0, fill=1)

    margin = 0.55 * inch
    top = H - margin
    left = margin
    right = W - margin

    # Header
    rr(left, top-70, right-left, 62, r=16, fill=card2)
    t(left+16, top-28, "DIAGNÓSTICO DE FINANZAS PERSONALES — REPORTE DE CONSULTORÍA", size=12.5, bold=True)
    edad = "—" if r.age is None else str(r.age)
    t(left+16, top-45, f"Consultor/a: {r.consultant}  |  Cliente: {r.client}", size=9.5, col=muted)
    t(left+16, top-60, f"Edad: {edad}  |  Ocupación: {r.occupation}  |  Dependientes: {r.dependents}", size=9.5, col=muted)
    tr(right-16, top-45, f"Fecha: {r.report_date}", size=9.5, col=muted)

    # Health pill
    pill_col = good if r.health_score >= 6 else (bad if r.health_score <= 3 else warn)
    c.setFillColor(colors.Color(pill_col.red, pill_col.green, pill_col.blue, alpha=0.18))
    c.setStrokeColor(line)
    c.roundRect(right-210, top-63, 190, 26, 13, stroke=1, fill=1)
    t(right-200, top-56, f"SALUD FINANCIERA: {r.health_label}", size=9.6, bold=True, col=pill_col)

    # KPI row (4 cards)
    kpi_y = top-150
    kpi_h = 58
    gap = 10
    kpi_w = (right-left - gap*3)/4

    def kpi(i, title, value, rating: Rating):
        x = left + i*(kpi_w+gap)
        rr(x, kpi_y, kpi_w, kpi_h, r=14, fill=card)
        t(x+12, kpi_y+kpi_h-18, title, size=8.4, col=muted)
        t(x+12, kpi_y+20, value, size=11.5, bold=True)
        t(x+12, kpi_y+6, f"{rating.emoji} {rating.label}", size=8.4, col=rating_color.get(rating.key, muted))

    kpi(0, "Flujo de caja mensual", fmt_pyg(r.cashflow), r.cashflow_rating)
    kpi(1, "Tasa de ahorro", fmt_pct(r.savings_rate), r.savings_rating)
    kpi(2, "Endeudamiento (DTI)", fmt_pct(r.dti), r.dti_rating)
    kpi(3, "Fondo de emergencia", f"{r.emergency_months:.1f} meses", r.emergency_rating)

    # Mid cards: perfil de riesgo (izq) + diagnóstico (der)
    mid_y = kpi_y - 210
    mid_h = 190
    lw = (right-left)*0.50-8
    rw = (right-left)*0.50-8
    rr(left, mid_y, lw, mid_h, r=18, fill=card)
    rr(left+lw+16, mid_y, rw, mid_h, r=18, fill=card)

    # Izquierda: perfil de riesgo
    t(left+16, mid_y+mid_h-24, "Perfil de riesgo", size=11.2, bold=True)
    t(left+16, mid_y+mid_h-40, f"{r.risk_category}  —  {r.risk_score}/{r.risk_max} pts", size=10, bold=True, col=accent)
    yy = para(left+16, mid_y+mid_h-58, r.risk_description, width_chars=52)

    bar_x = left+16
    bar_y = mid_y+30
    bar_w = lw-32
    bar_h = 14
    c.setFillColor(colors.Color(1, 1, 1, alpha=0.05))
    c.rect(bar_x, bar_y, bar_w, bar_h, stroke=0, fill=1)
    seg1 = bar_w * (r.risk_alloc_fixed/100.0)
    c.setFillColor(colors.Color(accent.red, accent.green, accent.blue, alpha=0.55))
    c.rect(bar_x, bar_y, seg1, bar_h, stroke=0, fill=1)
    c.setFillColor(colors.Color(warn.red, warn.green, warn.blue, alpha=0.55))
    c.rect(bar_x+seg1, bar_y, bar_w-seg1, bar_h, stroke=0, fill=1)
    t(left+16, bar_y-12, f"Mezcla ilustrativa: {r.risk_alloc_fixed}% bajo riesgo / {r.risk_alloc_variable}% mayor riesgo (no es recomendación)", size=7.6, col=muted)

    # Derecha: diagnóstico financiero
    rx = left+lw+16
    t(rx+16, mid_y+mid_h-24, "Diagnóstico financiero", size=11.2, bold=True)
    rows = [
        ("Ingreso mensual", fmt_pyg(r.income)),
        ("Gastos fijos", fmt_pyg(r.fixed_expenses)),
        ("Gastos variables", fmt_pyg(r.variable_expenses)),
        ("Cuota de deudas", fmt_pyg(r.debt_payment)),
        ("Ahorro/inversión mensual", fmt_pyg(r.savings_monthly)),
        ("Fondo de emergencia (monto)", fmt_pyg(r.emergency_fund)),
    ]
    if r.net_worth is not None:
        rows.append(("Patrimonio neto aprox.", fmt_pyg(r.net_worth)))
    ry = mid_y+mid_h-44
    for label, val in rows:
        t(rx+16, ry, label, size=8.9, col=muted)
        tr(rx+rw-16, ry, val, size=8.9, bold=True)
        ry -= 15

    # Bottom card: plan de acción
    bot_y = mid_y - 220
    bot_h = 200
    rr(left, bot_y, right-left, bot_h, r=18, fill=card)
    t(left+16, bot_y+bot_h-24, "Plan de acción sugerido", size=11.2, bold=True)
    yy = bot_y+bot_h-44
    for item in r.action_plan:
        for ln in wrap(item, 108):
            t(left+20, yy, ln, size=9.2, col=muted)
            yy -= 12.5
        yy -= 2
        if yy < bot_y + 14:
            break

    # Footer
    c.setFillColor(muted)
    c.setFont("Helvetica", 8.2)
    disclaimer_lines = wrap(DISCLAIMER, 100)
    brand_y = margin - 8 + (len(disclaimer_lines) - 1) * 10
    c.drawRightString(right, brand_y, "Consultoría de Finanzas Personales — Diplomado")
    for i, ln in enumerate(disclaimer_lines):
        c.drawString(left, margin-8-(i*10), ln)

    c.showPage()
    c.save()
    return buf.getvalue()
