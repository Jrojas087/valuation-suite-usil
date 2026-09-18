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

# El cuestionario de 10 preguntas mide sobre todo TOLERANCIA al riesgo (actitud/disposición
# psicológica frente a la volatilidad). No sustituye un análisis de CAPACIDAD de riesgo (qué
# tanta pérdida puede absorber el cliente según edad, estabilidad de ingresos, dependientes,
# horizonte y fondo de emergencia) — ambos conceptos son distintos en planificación financiera
# y deberían combinarse antes de sugerir una mezcla de inversión. Ver recomendación al docente.
RISK_METHOD_NOTE = (
    "Nota metodológica: este puntaje refleja principalmente la tolerancia al riesgo (actitud) del "
    "cliente. Antes de definir una mezcla de inversión, contrasta este resultado con su capacidad de "
    "riesgo real (edad, estabilidad de ingresos, dependientes y fondo de emergencia)."
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
    lines.append(RISK_METHOD_NOTE)
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
    lines.append(f"- Excedente antes de ahorro: {fmt_pyg(r.cashflow)}  [{r.cashflow_rating.emoji} {r.cashflow_rating.label}]")
    savings_txt = "No calculable (sin ingreso)" if r.savings_rating.key == "na" else fmt_pct(r.savings_rate)
    lines.append(f"- Tasa de ahorro: {savings_txt}  [{r.savings_rating.emoji} {r.savings_rating.label}]")
    dti_txt = "No calculable (sin ingreso)" if r.dti_rating.key == "na" else fmt_pct(r.dti)
    lines.append(f"- Endeudamiento (cuota/ingreso): {dti_txt}  [{r.dti_rating.emoji} {r.dti_rating.label}]")
    lines.append(f"- Fondo de emergencia: {r.emergency_months:.1f} meses cubiertos  [{r.emergency_rating.emoji} {r.emergency_rating.label}]")
    available_after_savings = r.cashflow - r.savings_monthly
    lines.append(f"- Disponible después de ahorro (excedente − ahorro declarado): {fmt_pyg(available_after_savings)}")
    if available_after_savings < 0:
        lines.append(
            "  ⚠️ El ahorro declarado supera el excedente disponible; verificar con el cliente el origen de "
            "esos fondos (otros ingresos, activos existentes o deuda adicional)."
        )
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

    def ellipsize(s, max_width, size=9.5, bold=False):
        # Trunca con "…" el texto que no entra en max_width (puntos), en vez de
        # dejar que invada lo que está dibujado a la derecha (fecha, píldora).
        font = "Helvetica-Bold" if bold else "Helvetica"
        if c.stringWidth(s, font, size) <= max_width:
            return s
        ell = "…"
        trimmed = s
        while trimmed and c.stringWidth(trimmed + ell, font, size) > max_width:
            trimmed = trimmed[:-1]
        return (trimmed.rstrip() + ell) if trimmed else ell

    # Background
    c.setFillColor(bg)
    c.rect(0, 0, W, H, stroke=0, fill=1)

    margin = 0.55 * inch
    top = H - margin
    left = margin
    right = W - margin

    # Header
    # Altura ampliada (62 -> 80) para poder mostrar el objetivo del cliente en una
    # línea propia; kpi_y se corre la misma cantidad (18pt) más abajo para que el
    # espaciado con la fila de KPIs no cambie.
    header_h = 80
    rr(left, top-88, right-left, header_h, r=16, fill=card2)
    t(left+16, top-28, "DIAGNÓSTICO DE FINANZAS PERSONALES — REPORTE DE CONSULTORÍA", size=12.5, bold=True)
    edad = "—" if r.age is None else str(r.age)

    # La píldora de salud financiera (dibujada más abajo) ocupa una franja vertical
    # que cruza tanto la fila 1 (Consultor/a | Cliente) como la fila 2 (Edad |
    # Ocupación | Dependientes), así que ambas deben frenar antes de su borde
    # izquierdo, no solo antes del texto de la fecha.
    pill_left_x = right-210
    date_str = f"Fecha: {r.report_date}"
    date_w = c.stringWidth(date_str, "Helvetica", 9.5)
    date_start_x = right-16 - date_w
    line1_max_w = min(date_start_x - 12, pill_left_x - 10) - (left+16)
    line1 = ellipsize(f"Consultor/a: {r.consultant}  |  Cliente: {r.client}", line1_max_w, size=9.5)
    t(left+16, top-45, line1, size=9.5, col=muted)
    tr(right-16, top-45, date_str, size=9.5, col=muted)

    line2_max_w = (pill_left_x - 10) - (left+16)
    line2 = ellipsize(
        f"Edad: {edad}  |  Ocupación: {r.occupation}  |  Dependientes: {r.dependents}", line2_max_w, size=9.5,
    )
    t(left+16, top-60, line2, size=9.5, col=muted)

    line3 = ellipsize(f"Objetivo: {r.objective}", (right-16) - (left+16), size=9.5)
    t(left+16, top-75, line3, size=9.5, col=accent)

    # Health pill
    pill_col = good if r.health_score >= 6 else (bad if r.health_score <= 3 else warn)
    c.setFillColor(colors.Color(pill_col.red, pill_col.green, pill_col.blue, alpha=0.18))
    c.setStrokeColor(line)
    c.roundRect(right-210, top-63, 190, 26, 13, stroke=1, fill=1)
    t(right-200, top-56, f"SALUD FINANCIERA: {r.health_label}", size=9.6, bold=True, col=pill_col)

    # KPI row (4 cards)
    kpi_y = top-168
    kpi_h = 58
    gap = 10
    kpi_w = (right-left - gap*3)/4

    def kpi(i, title, value, rating: Rating):
        x = left + i*(kpi_w+gap)
        rr(x, kpi_y, kpi_w, kpi_h, r=14, fill=card)
        t(x+12, kpi_y+kpi_h-18, title, size=8.4, col=muted)
        t(x+12, kpi_y+20, value, size=11.5, bold=True)
        t(x+12, kpi_y+6, f"{rating.emoji} {rating.label}", size=8.4, col=rating_color.get(rating.key, muted))

    savings_val = "—" if r.savings_rating.key == "na" else fmt_pct(r.savings_rate)
    dti_val = "—" if r.dti_rating.key == "na" else fmt_pct(r.dti)
    emergency_val = "—" if r.emergency_rating.key == "na" else f"{r.emergency_months:.1f} meses"
    kpi(0, "Excedente antes de ahorro", fmt_pyg(r.cashflow), r.cashflow_rating)
    kpi(1, "Tasa de ahorro", savings_val, r.savings_rating)
    kpi(2, "Endeudamiento (DTI)", dti_val, r.dti_rating)
    kpi(3, "Fondo de emergencia", emergency_val, r.emergency_rating)

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
    yy = para(left+16, yy - 4, RISK_METHOD_NOTE, width_chars=60, size=7.6, leading=9.5, col=accent)

    bar_x = left+16
    bar_h = 14
    caption_text = (
        f"Mezcla ilustrativa: {r.risk_alloc_fixed}% bajo riesgo / {r.risk_alloc_variable}% mayor riesgo "
        f"(no es recomendación)"
    )
    caption_leading = 9.2
    caption_lines = wrap(caption_text, 58)
    # Espacio que necesita la leyenda debajo de la barra: si no entra en una sola
    # línea (tarjeta angosta / texto más largo de lo habitual), reserva 2 líneas en
    # vez de dejar que se salga de la tarjeta.
    caption_block_h = 12 + (len(caption_lines) - 1) * caption_leading
    # La posición de la barra se ajusta al contenido de arriba (descripción + nota
    # metodológica) para no solaparse si el texto ocupa más líneas de lo habitual,
    # y deja piso suficiente para que la leyenda (una o dos líneas) no se salga por
    # abajo de la tarjeta.
    bar_y = max(mid_y + 6 + caption_block_h, min(mid_y + 30, yy - 12))
    bar_w = lw-32
    c.setFillColor(colors.Color(1, 1, 1, alpha=0.05))
    c.rect(bar_x, bar_y, bar_w, bar_h, stroke=0, fill=1)
    seg1 = bar_w * (r.risk_alloc_fixed/100.0)
    c.setFillColor(colors.Color(accent.red, accent.green, accent.blue, alpha=0.55))
    c.rect(bar_x, bar_y, seg1, bar_h, stroke=0, fill=1)
    c.setFillColor(colors.Color(warn.red, warn.green, warn.blue, alpha=0.55))
    c.rect(bar_x+seg1, bar_y, bar_w-seg1, bar_h, stroke=0, fill=1)
    cap_y = bar_y - 12
    for ln in caption_lines:
        t(left+16, cap_y, ln, size=7.6, col=muted)
        cap_y -= caption_leading

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
    available_after_savings = r.cashflow - r.savings_monthly
    rows.append(("Disponible después de ahorro", fmt_pyg(available_after_savings)))
    ry = mid_y+mid_h-44
    for label, val in rows:
        t(rx+16, ry, label, size=8.9, col=muted)
        val_col = bad if (label == "Disponible después de ahorro" and available_after_savings < 0) else text
        tr(rx+rw-16, ry, val, size=8.9, bold=True, col=val_col)
        ry -= 15

    # Bottom card: plan de acción
    # Usa todo el espacio vertical libre entre las tarjetas del medio y el pie de
    # página (antes quedaba una franja en blanco y una tarjeta de altura fija que
    # truncaba planes de acción largos sin avisar).
    footer_reserved = 34  # línea de marca + disclaimer (ver footer más abajo)
    bot_top = mid_y - 20
    bot_y = margin + footer_reserved
    bot_h = bot_top - bot_y
    rr(left, bot_y, right-left, bot_h, r=18, fill=card)
    t(left+16, bot_y+bot_h-24, "Plan de acción sugerido", size=11.2, bold=True)
    yy = bot_y+bot_h-44
    truncated = False
    items = list(r.action_plan)
    for idx, item in enumerate(items):
        wrapped_lines = wrap(item, 108)
        stop = False
        for line_idx, ln in enumerate(wrapped_lines):
            # Antes de dibujar cada línea (no recién al terminar el ítem) chequeamos si
            # queda contenido pendiente y ya no hay lugar: si dibujáramos igual, la última
            # línea del ítem podía terminar superpuesta con el aviso de continuación.
            more_content = (line_idx < len(wrapped_lines) - 1) or (idx < len(items) - 1)
            if yy < bot_y + 24 and more_content:
                truncated = True
                stop = True
                break
            t(left+20, yy, ln, size=9.2, col=muted)
            yy -= 12.5
        if stop:
            break
        yy -= 2
    if truncated:
        t(left+20, bot_y+12, "(continúa en el reporte TXT — ver ítems adicionales)", size=8.2, col=accent)

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
