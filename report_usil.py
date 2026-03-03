# report_usil.py
# Exporter premium (TXT + PDF) — sin matplotlib, PDF 100% ReportLab

from __future__ import annotations

import io
import math
import textwrap
from dataclasses import dataclass
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
    return f"{x*100:.2f}%"

def fmt_pyg(x: Optional[float]) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    return "Gs. {:,.0f}".format(float(x)).replace(",", ".")

def wrap(s: str, width: int) -> list[str]:
    return textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False)

def badge(verdict: str) -> str:
    return {"APROBADO": "✅", "OBSERVADO": "⚠️", "RECHAZADO": "⛔"}.get(verdict, "—")


# ----------------------------
# Data model (sin defaults “antes” de non-default)
# ----------------------------
@dataclass
class OnePager:
    institution: str
    program: str
    course: str
    currency: str
    project: str
    responsible: str
    report_date: str

    verdict: str
    rationale: str

    # KPIs
    npv_base: Optional[float]
    irr_base: Optional[float]
    payback_simple: Optional[float]
    payback_discounted: Optional[float]

    # Assumptions
    n_years: int
    g_exp: float
    g_inf: float
    wacc: float
    ke: float
    kd: float
    capex0: float
    fcf1: float

    # Bridge
    ebit: float
    nopat: float
    delta_wc: float
    capex_y1: float

    # Components
    pv_fcf: Optional[float]
    pv_tv: Optional[float]
    tv: Optional[float]

    # Monte Carlo
    sims: int
    valid_rate: Optional[float]
    prob_neg: Optional[float]
    p5: Optional[float]
    p50: Optional[float]
    p95: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    cvar5: Optional[float]

    checks: Sequence[Tuple[str, bool]]
    fcf_years: Sequence[float]


# ----------------------------
# TXT one-pager (para rápida descarga)
# ----------------------------
def build_onepager_text(r: OnePager) -> str:
    lines: list[str] = []
    lines.append("ONE-PAGER EJECUTIVO — EVALUACIÓN FINANCIERA")
    lines.append(f"{r.institution} — {r.program} — {r.course}")
    lines.append(f"Moneda: {r.currency}")
    lines.append(f"Fecha: {r.report_date}")
    lines.append(f"Proyecto: {r.project} | Responsable: {r.responsible}")
    lines.append("")
    lines.append(f"DICTAMEN: {r.verdict} {badge(r.verdict)}")
    lines.append(r.rationale)
    lines.append("")
    lines.append("KPIs (determinístico)")
    lines.append(f"- VAN (base): {fmt_pyg(r.npv_base)}")
    lines.append(f"- TIR (base): {fmt_pct(r.irr_base)}")
    pb = "N/A" if r.payback_simple is None else f"{r.payback_simple:.2f} años"
    pbd = "N/A" if r.payback_discounted is None else f"{r.payback_discounted:.2f} años"
    lines.append(f"- Payback: {pb} | Payback descontado: {pbd}")
    lines.append("")
    lines.append("Supuestos clave")
    lines.append(f"- Horizonte: {r.n_years} años + perpetuidad")
    lines.append(f"- g explícito: {fmt_pct(r.g_exp)} | g∞: {fmt_pct(r.g_inf)}")
    lines.append(f"- WACC: {fmt_pct(r.wacc)} (Ke {fmt_pct(r.ke)} | Kd {fmt_pct(r.kd)})")
    lines.append(f"- CAPEX₀: {fmt_pyg(r.capex0)} | FCF1: {fmt_pyg(r.fcf1)}")
    lines.append("")
    lines.append("Flujos proyectados (FCF) — ciclo 1..N (+ TV separado)")
    for i, f in enumerate(r.fcf_years, start=1):
        lines.append(f"- Año {i}: {fmt_pyg(f)}")
    lines.append(f"- TV (en año {r.n_years}): {fmt_pyg(r.tv)}")
    lines.append("")
    lines.append("Puente contable → FCF (Año 1)")
    lines.append(f"- EBIT: {fmt_pyg(r.ebit)} | NOPAT: {fmt_pyg(r.nopat)}")
    lines.append(f"- ΔCT (AR+INV−AP): {fmt_pyg(r.delta_wc)} | CAPEX Año 1: {fmt_pyg(r.capex_y1)}")
    lines.append("")
    lines.append("Riesgo (Monte Carlo)")
    lines.append(f"- Simulaciones: {r.sims:,} | válidas: {('—' if r.valid_rate is None else f'{r.valid_rate*100:.1f}%')}")
    lines.append(f"- P(VAN<0): {('—' if r.prob_neg is None else f'{r.prob_neg*100:.1f}%')}")
    lines.append(f"- P5: {fmt_pyg(r.p5)} | P50: {fmt_pyg(r.p50)} | P95: {fmt_pyg(r.p95)}")
    lines.append(f"- Media: {fmt_pyg(r.mean)} | σ: {fmt_pyg(r.std)} | CVaR5: {fmt_pyg(r.cvar5)}")
    lines.append("")
    lines.append("Checklist Comité (automático)")
    for label, ok in r.checks:
        lines.append(("✅ " if ok else "❌ ") + label)
    lines.append("")
    lines.append("Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence.")
    return "\n".join(lines)


# ----------------------------
# PDF premium (one page) — ReportLab
# ----------------------------
def generate_onepager_pdf(
    onepager: OnePager,
    hist_counts=None,
    hist_edges=None,
) -> bytes:
    """
    Exporta el One-Pager en formato "Blue Dashboard" (A4) usando ReportLab.
    - Mantiene la firma original para no romper la app.
    - Usa `onepager.fcf_years` para el gráfico de flujos.
    - Usa `hist_counts` (y opcionalmente `hist_edges`) para el histograma Monte Carlo.
    """
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab no está disponible. Agregar `reportlab` a requirements.txt.")

    # Imports extra (reportlab.graphics)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.graphics.shapes import Drawing, Rect
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.graphics import renderPDF
    except Exception as e:
        raise RuntimeError("Faltan módulos de ReportLab (graphics). Verificar instalación de `reportlab`.") from e

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    # -------------------------
    # Paleta (alineada al modelo)
    # -------------------------
    def hexcolor(hx: str):
        hx = hx.lstrip("#")
        return colors.Color(int(hx[0:2], 16) / 255, int(hx[2:4], 16) / 255, int(hx[4:6], 16) / 255)

    BG_TOP = hexcolor("#0A1630")
    BG_BOT = hexcolor("#081024")
    CARD = hexcolor("#0E2A47")
    CARD2 = hexcolor("#0C2540")
    BORDER = hexcolor("#2B4F73")
    DIV = hexcolor("#1C3C5E")
    TEXT = colors.whitesmoke
    MUTED = hexcolor("#B8C7D9")
    ACCENT = hexcolor("#62B6FF")
    ACCENT2 = hexcolor("#2AD3A6")

    # -------------------------
    # Background (gradiente)
    # -------------------------
    steps = 160
    for i in range(steps):
        t = i / (steps - 1)
        col = colors.Color(
            BG_TOP.red * (1 - t) + BG_BOT.red * t,
            BG_TOP.green * (1 - t) + BG_BOT.green * t,
            BG_TOP.blue * (1 - t) + BG_BOT.blue * t,
        )
        y0 = H * (i / steps)
        c.setFillColor(col)
        c.rect(0, y0, W, H / steps + 1, stroke=0, fill=1)

    # -------------------------
    # Helpers UI
    # -------------------------
    def round_card(x, y, w, h, title=None, icon=False):
        c.setFillColor(CARD)
        c.setStrokeColor(BORDER)
        c.setLineWidth(1)
        c.roundRect(x, y, w, h, 10, stroke=1, fill=1)
        if title:
            if icon:
                cx, cy = x + 12, y + h - 18
                c.setFillColor(ACCENT)
                c.saveState()
                c.translate(cx, cy)
                c.rotate(45)
                c.rect(-3.5, -3.5, 7, 7, stroke=0, fill=1)
                c.restoreState()
            c.setFillColor(TEXT)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x + (24 if icon else 12), y + h - 22, title)
            c.setStrokeColor(DIV)
            c.line(x + 10, y + h - 30, x + w - 10, y + h - 30)

    def badge(x, y, text):
        w, h = 50 * mm, 6 * mm
        c.setFillColor(hexcolor("#113A56"))
        c.setStrokeColor(hexcolor("#1F6E5C"))
        c.setLineWidth(1)
        c.roundRect(x, y, w, h, 6, stroke=1, fill=1)
        c.setFillColor(ACCENT2)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawCentredString(x + w / 2, y + 1.8 * mm, text)

    # -------------------------
    # Header
    # -------------------------
    header_h = 30 * mm
    margin = 12 * mm
    header_y = H - margin - header_h

    c.setFillColor(CARD2)
    c.setStrokeColor(BORDER)
    c.setLineWidth(1)
    c.roundRect(margin, header_y, W - 2 * margin, header_h, 12, stroke=1, fill=1)

    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin + 6 * mm, header_y + header_h - 9 * mm, "ONE-PAGER EJECUTIVO — EVALUACIÓN FINANCIERA (PYG)")

    c.setFont("Helvetica", 9)
    c.setFillColor(MUTED)
    c.drawString(margin + 6 * mm, header_y + header_h - 16 * mm, f"{onepager.institution} — {onepager.program}")
    c.drawString(margin + 6 * mm, header_y + header_h - 21 * mm, f"Proyecto {onepager.project} — Docente Responsable: {onepager.responsible}")

    badge(W - margin - 50 * mm - 6 * mm, header_y + 4 * mm, f"DICTAMEN: {onepager.verdict}")

    # -------------------------
    # Layout grid 2 columnas x 3 filas
    # -------------------------
    gap = 8 * mm
    content_top = header_y - gap
    content_bottom = margin
    content_h = content_top - content_bottom

    col_w = (W - 2 * margin - gap) / 2
    left_x = margin
    right_x = margin + col_w + gap

    r1 = 58 * mm
    r2 = 64 * mm
    r3 = content_h - r1 - r2 - 2 * gap

    y1 = content_top - r1
    y2 = y1 - gap - r2
    y3 = content_bottom

    # -------------------------
    # Indicadores (tabla sin "Enfoque")
    # -------------------------
    round_card(left_x, y1, col_w, r1, "Indicadores Clave", icon=True)

    tx = left_x + 12
    tw = col_w - 24
    row_h = 9.5 * mm
    ty = y1 + r1 - 46

    c.setFillColor(hexcolor("#113556"))
    c.setStrokeColor(DIV)
    c.roundRect(tx, ty, tw, row_h, 6, stroke=1, fill=1)
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(tx + 8, ty + 3.2 * mm, "Métrica")
    c.drawRightString(tx + tw - 8, ty + 3.2 * mm, "Resultado")

    pb = "N/A" if onepager.payback_simple is None else f"{onepager.payback_simple:.2f} años"
    pbd = "N/A" if onepager.payback_discounted is None else f"{onepager.payback_discounted:.2f} años"
    prob = "—" if onepager.prob_neg is None else f"{onepager.prob_neg*100:.1f}%"

    indicadores = [
        ("VAN (Base)", fmt_pyg(onepager.npv_base)),
        ("TIR (Base)", fmt_pct(onepager.irr_base)),
        ("Payback Simple", pb),
        ("Payback Descontado", pbd),
        ("P(VAN < 0)", prob),
    ]

    for i, (mtr, res) in enumerate(indicadores):
        ry = ty - (i + 1) * row_h
        c.setFillColor(hexcolor("#0B223B"))
        c.setStrokeColor(DIV)
        c.roundRect(tx, ry, tw, row_h, 6, stroke=1, fill=1)
        c.setFillColor(TEXT)
        c.setFont("Helvetica-Bold" if i < 2 else "Helvetica", 9)
        c.drawString(tx + 8, ry + 3.1 * mm, mtr)
        c.drawRightString(tx + tw - 8, ry + 3.1 * mm, res)

    # -------------------------
    # Estructura del valor
    # -------------------------
    round_card(right_x, y1, col_w, r1, "Estructura del valor")

    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawString(right_x + 12, y1 + r1 - 46, "VAN = PV(FCF 1..N) + PV(TV) – CAPEX")

    pv_fcf = onepager.pv_fcf
    pv_tv = onepager.pv_tv
    capex0 = onepager.capex0
    vals = [
        ("PV FCF:", fmt_pyg(pv_fcf)),
        ("PV TV:", fmt_pyg(pv_tv)),
        ("CAPEX:", f"({fmt_pyg(capex0)})" if capex0 is not None else "—"),
    ]

    base_y = y1 + r1 - 62
    for i, (k, v) in enumerate(vals):
        yy = base_y - i * 11 * mm
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 10)
        c.drawString(right_x + 12, yy, k)
        c.setFillColor(TEXT)
        c.setFont("Helvetica-Bold", 10)
        c.drawRightString(right_x + col_w - 12, yy, v)

    c.setStrokeColor(DIV)
    c.line(right_x + 12, y1 + 18 * mm, right_x + col_w - 12, y1 + 18 * mm)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(right_x + 12, y1 + 9 * mm, "VAN Base:")
    c.drawRightString(right_x + col_w - 12, y1 + 9 * mm, fmt_pyg(onepager.npv_base))

    # -------------------------
    # Decisión & Supuestos
    # -------------------------
    round_card(left_x, y2, col_w, r2, "Decisión & Supuestos", icon=True)

    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left_x + 12, y2 + r2 - 46, "Conclusión Ejecutiva")

    # Rationale (máx 3 líneas)
    c.setFillColor(TEXT)
    c.setFont("Helvetica", 9)
    yy = y2 + r2 - 58
    for ln in wrap(onepager.rationale or "", 62)[:3]:
        c.drawString(left_x + 12, yy, ln)
        yy -= 11

    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left_x + 12, y2 + r2 - 98, "Supuestos Clave")

    bullets = [
        f"Horizonte: {onepager.n_years} años + perpetuidad",
        f"g explícito: {fmt_pct(onepager.g_exp)}  |  g perpetuo: {fmt_pct(onepager.g_inf)}",
        f"WACC: {fmt_pct(onepager.wacc)}  —  Ke: {fmt_pct(onepager.ke)}  |  Kd: {fmt_pct(onepager.kd)}",
        f"CAPEX: {fmt_pyg(onepager.capex0)}",
    ]

    c.setFont("Helvetica", 9)
    for i, tline in enumerate(bullets):
        yb = y2 + r2 - 112 - i * 10.3
        c.setFillColor(ACCENT)
        c.circle(left_x + 14, yb + 3, 1.2, stroke=0, fill=1)
        c.setFillColor(TEXT)
        c.drawString(left_x + 20, yb, tline)

    # -------------------------
    # Retorno del Valor
    # -------------------------
    round_card(right_x, y2, col_w, r2, "Retorno del Valor", icon=True)

    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawString(right_x + 12, y2 + r2 - 46, "VAN = PV(FCF 1..N) + PV(TV) – CAPEX")

    base_y = y2 + r2 - 62
    for i, (k, v) in enumerate(vals):
        yy = base_y - i * 11 * mm
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 10)
        c.drawString(right_x + 12, yy, k)
        c.setFillColor(TEXT)
        c.setFont("Helvetica-Bold", 10)
        c.drawRightString(right_x + col_w - 12, yy, v)

    c.setStrokeColor(DIV)
    c.line(right_x + 12, y2 + 18 * mm, right_x + col_w - 12, y2 + 18 * mm)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(right_x + 12, y2 + 9 * mm, "VAN Base:")
    c.drawRightString(right_x + col_w - 12, y2 + 9 * mm, fmt_pyg(onepager.npv_base))

    # -------------------------
    # Flujos Proyectados (barras)
    # -------------------------
    round_card(left_x, y3, col_w, r3, "Flujos Proyectados (Ciclo 1..N) + TV", icon=True)

    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawString(left_x + 12, y3 + r3 - 46, "FCF por ciclo (TV reportado separadamente)")

    fcf_vals = list(onepager.fcf_years or [])
    if not fcf_vals:
        fcf_vals = [0, 0, 0, 0, 0]
    cats = [str(i + 1) for i in range(len(fcf_vals))]

    # Normalización simple para el gráfico (solo visual)
    maxv = max(fcf_vals) if max(fcf_vals) > 0 else 1.0
    # Si son valores muy grandes (Gs.), escalamos a "miles de millones" para que el eje tenga sentido
    scale = 1.0
    if maxv >= 1e9:
        scale = 1e9
        fcf_plot = [v / scale for v in fcf_vals]
    else:
        fcf_plot = fcf_vals

    dw = Drawing(col_w - 24, r3 - 78)
    bc = VerticalBarChart()
    bc.x = 20
    bc.y = 18
    bc.height = max(40, r3 - 110)
    bc.width = col_w - 70
    bc.data = [fcf_plot]
    bc.categoryAxis.categoryNames = cats
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = max(fcf_plot) * 1.15 if max(fcf_plot) > 0 else 1
    bc.valueAxis.valueStep = max(bc.valueAxis.valueMax / 6.0, 1)
    bc.valueAxis.labels.fontName = "Helvetica"
    bc.valueAxis.labels.fontSize = 8
    bc.categoryAxis.labels.fontName = "Helvetica"
    bc.categoryAxis.labels.fontSize = 8
    bc.strokeColor = DIV
    bc.bars[0].fillColor = ACCENT
    bc.barWidth = 12
    dw.add(bc)
    renderPDF.draw(dw, c, left_x + 12, y3 + 20)

    tv_text = "—" if onepager.tv is None else fmt_pyg(onepager.tv)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8.8)
    c.drawString(left_x + 12, y3 + 10, f"TV en ciclo {onepager.n_years} • {tv_text}")

    # -------------------------
    # Riesgo — Simulación Monte Carlo (histograma)
    # -------------------------
    round_card(right_x, y3, col_w, r3, "Riesgo — Simulación Monte Carlo", icon=True)

    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawString(right_x + 12, y3 + r3 - 46, f"Simulaciones: {onepager.sims:,}".replace(",", "."))
    vr = "—" if onepager.valid_rate is None else f"{onepager.valid_rate*100:.1f}%".replace(".", ",")
    c.drawString(right_x + 12, y3 + r3 - 58, f"Válidas: {vr}")
    c.drawString(right_x + 12, y3 + r3 - 70, f"P(VAN < 0): {prob.replace('.', ',')}")

    # Histograma
    hist_w = col_w - 24
    hist_h = 52 * mm
    d2 = Drawing(hist_w, hist_h)

    counts = hist_counts
    if counts is None or len(counts) == 0:
        # fallback visual
        counts = [2, 4, 7, 10, 12, 11, 9, 7, 5, 3, 2]

    maxb = max(counts) if max(counts) > 0 else 1
    x0, y0 = 10, 8
    bw = (hist_w - 20) / len(counts)
    for i, v in enumerate(counts):
        hh = (v / maxb) * (hist_h - 20)
        d2.add(Rect(x0 + i * bw + 1, y0, bw - 2, hh, fillColor=ACCENT, strokeColor=None))
    d2.add(Rect(x0, y0, hist_w - 20, 1, fillColor=DIV, strokeColor=None))
    renderPDF.draw(d2, c, right_x + 12, y3 + r3 - 46 - hist_h - 10)

    # Distribución (más chica y más abajo)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(right_x + 12, y3 + 44, "Distribución")

    dist = [
        ("P5:", fmt_pyg(onepager.p5)),
        ("P50:", fmt_pyg(onepager.p50)),
        ("P95:", fmt_pyg(onepager.p95)),
    ]
    for i, (k, v) in enumerate(dist):
        yy = y3 + 34 - i * 10.5
        c.setFont("Helvetica", 8.5)
        c.setFillColor(MUTED)
        c.drawString(right_x + 12, yy, k)
        c.setFont("Helvetica-Bold", 8.5)
        c.setFillColor(TEXT)
        c.drawString(right_x + 36, yy, v)

    c.setStrokeColor(DIV)
    c.line(right_x + 12, y3 + 28, right_x + col_w - 12, y3 + 28)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(right_x + 12, y3 + 14, "VAN Base:")
    c.drawRightString(right_x + col_w - 12, y3 + 14, fmt_pyg(onepager.npv_base))

    c.showPage()
    c.save()
    return buf.getvalue()

