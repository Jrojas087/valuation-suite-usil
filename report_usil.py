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
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab no está disponible. Agregar `reportlab` a requirements.txt.")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter

    # Paleta
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

    # helpers
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

    def para(x, y, s, width_chars=90, size=9.6, leading=12, col=muted):
        yy = y
        for ln in wrap(s, width_chars):
            t(x, yy, ln, size=size, bold=False, col=col)
            yy -= leading
        return yy

    # Background
    c.setFillColor(bg)
    c.rect(0, 0, W, H, stroke=0, fill=1)

    # aumentar márgenes superiores/inferiores para evitar solapamiento en PDF
    margin = 0.8 * inch
    top = H - margin
    left = margin
    right = W - margin

    # Header (elevado ligeramente para más espacio superior)
    rr(left, top-80, right-left, 72, r=16, fill=card2)
    t(left+16, top-34, "ONE-PAGER EJECUTIVO — EVALUACIÓN FINANCIERA (PYG)", size=13, bold=True)
    t(left+16, top-52, f"{onepager.institution} — {onepager.program} — {onepager.course}", size=9.5, col=muted)
    t(left+16, top-70, f"Proyecto: {onepager.project}  |  Responsable: {onepager.responsible}", size=9.5, col=muted)
    tr(right-16, top-52, f"Fecha: {onepager.report_date}", size=9.5, col=muted)

    # Verdict pill
    pill_col = good if onepager.verdict=="APROBADO" else (bad if onepager.verdict=="RECHAZADO" else warn)
    c.setFillColor(colors.Color(pill_col.red, pill_col.green, pill_col.blue, alpha=0.18))
    c.setStrokeColor(line)
    c.roundRect(right-170, top-63, 150, 26, 13, stroke=1, fill=1)
    t(right-160, top-56, f"DICTAMEN: {onepager.verdict}", size=10.5, bold=True, col=pill_col)

    # KPI row (6 cards)
    # elevar KPI row un poco si la cabecera aumentó su altura
    kpi_y = top-160
    kpi_h = 54
    gap = 10
    kpi_w = (right-left - gap*5)/6

    def kpi(i, title, value, sub):
        x = left + i*(kpi_w+gap)
        rr(x, kpi_y, kpi_w, kpi_h, r=14, fill=card)
        t(x+12, kpi_y+kpi_h-18, title, size=8.6, col=muted)
        t(x+12, kpi_y+18, value, size=11.5, bold=True)
        t(x+12, kpi_y+6, sub, size=8.3, col=muted)

    pb = "N/A" if onepager.payback_simple is None else f"{onepager.payback_simple:.2f} a"
    pbd = "N/A" if onepager.payback_discounted is None else f"{onepager.payback_discounted:.2f} a"
    prob = "—" if onepager.prob_neg is None else f"{onepager.prob_neg*100:.1f}%"

    kpi(0, "VAN (base)", fmt_pyg(onepager.npv_base), "Determinístico")
    kpi(1, "TIR (base)", fmt_pct(onepager.irr_base), "Determinístico")
    kpi(2, "Payback", pb, "Simple")
    kpi(3, "Payback (desc.)", pbd, "Descontado")
    kpi(4, "P(VAN<0)", prob, "Monte Carlo")
    kpi(5, "P50 (VAN)", fmt_pyg(onepager.p50), "Monte Carlo")

    # Mid cards
    mid_y = kpi_y - 220
    mid_h = 200
    rr(left, mid_y, (right-left)*0.55-8, mid_h, r=18, fill=card)
    rr(left+(right-left)*0.55+8, mid_y, (right-left)*0.45-8, mid_h, r=18, fill=card)

    # Left mid: decision & assumptions
    t(left+16, mid_y+mid_h-24, "Decisión & supuestos", size=11.2, bold=True)
    yy = para(left+16, mid_y+mid_h-42, onepager.rationale, width_chars=70)
    t(left+16, yy-8, "Supuestos clave", size=10.2, bold=True, col=text)
    yy -= 24
    bullets = [
        f"Horizonte: {onepager.n_years} años + perpetuidad",
        f"g explícito: {fmt_pct(onepager.g_exp)} | g∞: {fmt_pct(onepager.g_inf)}",
        f"WACC: {fmt_pct(onepager.wacc)} (Ke {fmt_pct(onepager.ke)} | Kd {fmt_pct(onepager.kd)})",
        f"CAPEX₀: {fmt_pyg(onepager.capex0)} | FCF1: {fmt_pyg(onepager.fcf1)}",
    ]
    for b in bullets:
        t(left+20, yy, "• " + b, size=9.4, col=muted)
        yy -= 12

    # Right mid: value bridge
    rx = left+(right-left)*0.55+8
    rw = (right-left)*0.45-8
    t(rx+16, mid_y+mid_h-24, "Estructura del valor", size=11.2, bold=True)
    t(rx+16, mid_y+mid_h-40, "VAN = PV(FCF 1..N) + PV(TV) − CAPEX₀ (TV separado)", size=8.8, col=muted)

    # Draw bridge bar
    bar_x = rx+16
    bar_y = mid_y+78
    bar_w = rw-32
    bar_h = 16
    pv_fcf = onepager.pv_fcf or 0.0
    pv_tv = onepager.pv_tv or 0.0
    capex0 = onepager.capex0 or 0.0
    total_pos = max(pv_fcf + pv_tv, 1e-9)

    # background bar
    c.setFillColor(colors.Color(1,1,1,alpha=0.05))
    c.rect(bar_x, bar_y, bar_w, bar_h, stroke=0, fill=1)

    # segments
    seg_fcf = bar_w * (pv_fcf/total_pos)
    seg_tv = bar_w * (pv_tv/total_pos)
    c.setFillColor(colors.Color(accent.red, accent.green, accent.blue, alpha=0.55))
    c.rect(bar_x, bar_y, seg_fcf, bar_h, stroke=0, fill=1)
    c.setFillColor(colors.Color(accent.red, accent.green, accent.blue, alpha=0.30))
    c.rect(bar_x+seg_fcf, bar_y, seg_tv, bar_h, stroke=0, fill=1)

    t(rx+16, mid_y+58, f"PV FCF: {fmt_pyg(onepager.pv_fcf)}", size=9.1, col=muted)
    t(rx+16, mid_y+44, f"PV TV:  {fmt_pyg(onepager.pv_tv)}", size=9.1, col=muted)
    t(rx+16, mid_y+30, f"CAPEX₀: {fmt_pyg(-capex0)}", size=9.1, col=muted)
    t(rx+16, mid_y+12, f"VAN (base): {fmt_pyg(onepager.npv_base)}", size=11.0, bold=True, col=text)

    # Bottom row: FCF chart + Monte Carlo histogram
    bot_y = mid_y - 230
    bot_h = 210
    rr(left, bot_y, (right-left)*0.55-8, bot_h, r=18, fill=card)
    rr(left+(right-left)*0.55+8, bot_y, (right-left)*0.45-8, bot_h, r=18, fill=card)

    # Bottom-left: FCF bars
    t(left+16, bot_y+bot_h-24, "Flujos proyectados (ciclo 1..N) + TV", size=11.2, bold=True)
    t(left+16, bot_y+bot_h-40, "FCF por ciclo (TV se reporta separado)", size=8.8, col=muted)

    bx = left+16
    by = bot_y+48
    bw = (right-left)*0.55-8-32
    bh = 120
    f = list(onepager.fcf_years)
    maxf = max(f) if f else 1.0
    nb = len(f)
    if nb:
        gapb = 6
        barw = (bw - gapb*(nb-1))/nb
        for i, val in enumerate(f):
            h = (val/maxf) * bh if maxf > 0 else 0
            x = bx + i*(barw+gapb)
            c.setFillColor(colors.Color(accent.red, accent.green, accent.blue, alpha=0.55))
            c.rect(x, by, barw, h, stroke=0, fill=1)
            c.setFillColor(muted)
            c.setFont("Helvetica", 7.8)
            c.drawCentredString(x+barw/2, by-10, str(i+1))
    t(left+16, bot_y+26, f"TV (en ciclo {onepager.n_years}): {fmt_pyg(onepager.tv)}", size=9.1, col=muted)

    # Bottom-right: Monte Carlo histogram mini
    mx = left+(right-left)*0.55+8
    mw = (right-left)*0.45-8
    t(mx+16, bot_y+bot_h-24, "Riesgo (Monte Carlo) — lectura ejecutiva", size=11.2, bold=True)

    line1 = f"Simulaciones: {onepager.sims:,} | válidas: {('—' if onepager.valid_rate is None else f'{onepager.valid_rate*100:.1f}%')} | P(VAN<0): {('—' if onepager.prob_neg is None else f'{onepager.prob_neg*100:.1f}%')}"
    line2 = f"P5/P50/P95: {fmt_pyg(onepager.p5)} / {fmt_pyg(onepager.p50)} / {fmt_pyg(onepager.p95)}"
    line3 = f"Media: {fmt_pyg(onepager.mean)} | σ: {fmt_pyg(onepager.std)} | CVaR5: {fmt_pyg(onepager.cvar5)}"
    t(mx+16, bot_y+bot_h-42, line1, size=8.7, col=muted)
    t(mx+16, bot_y+bot_h-56, line2, size=8.7, col=muted)
    t(mx+16, bot_y+bot_h-70, line3, size=8.7, col=muted)

    # histogram draw area
    hx = mx+16
    hy = bot_y+50
    hw = mw-32
    hh = 110

    c.setFillColor(colors.Color(1,1,1,alpha=0.05))
    c.rect(hx, hy, hw, hh, stroke=0, fill=1)

    if hist_counts is not None and hist_edges is not None and len(hist_counts) > 0:
        maxc = max(hist_counts) if max(hist_counts) > 0 else 1
        nb = len(hist_counts)
        barw = hw/nb
        for i, cnt in enumerate(hist_counts):
            h = (cnt/maxc)*hh
            x = hx + i*barw
            c.setFillColor(colors.Color(accent.red, accent.green, accent.blue, alpha=0.55))
            c.rect(x, hy, barw*0.90, h, stroke=0, fill=1)

        # markers P5/P50/P95
        def x_at(value):
            lo = hist_edges[0]
            hi = hist_edges[-1]
            if hi <= lo:
                return hx
            return hx + (float(value)-lo)/(hi-lo)*hw

        for v in [onepager.p5, onepager.p50, onepager.p95]:
            if v is None or not math.isfinite(v):
                continue
            xx = x_at(v)
            c.setStrokeColor(colors.Color(234/255,241/255,1,alpha=0.60))
            c.setLineWidth(1.2)
            c.setDash(4, 4)
            c.line(xx, hy, xx, hy+hh)
            c.setDash()

    # Footer (más separación desde el borde)
    c.setFillColor(muted)
    c.setFont("Helvetica", 8.4)
    c.drawString(left, margin-12, "Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence.")
    c.drawRightString(right, margin-12, "ValuationSuite USIL (PYG)")

    c.showPage()
    c.save()
    return buf.getvalue()
