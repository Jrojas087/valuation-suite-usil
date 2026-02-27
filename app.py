# -*- coding: utf-8 -*-
"""
ValuationSuite USIL (PYG) — Streamlit App
Marco: DCF + Monte Carlo (incertidumbre razonable) + One-Pager ejecutivo (PDF opcional)
Autor: (generado) — Ajustado para Streamlit Cloud (sin matplotlib; PDF con ReportLab)

✅ Ajustes solicitados:
- Moneda fija PYG (Gs.)
- Sin selectores "base de medición" (solo nominal) ni D/E (solo contable)
- Sin selector de semilla (RNG interno)
- Comité por default: VAN base > 0, P(VAN<0) ≤ 20%, P50>0, P5>0
- Flujos explícitos ciclo 1..N + g perpetuidad (TV separado)
- Puente contable → FCF Año 1 (ventas, costos, depreciación, CAPEX, CT por componentes)
- KPIs: VAN, TIR, Payback simple y descontado
- Dashboard moderno: tiles + gráficos (Flujos, Puente de valor, Distribución Monte Carlo + percentiles)
"""

from __future__ import annotations

import io
import math
import textwrap
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Optional PDF (ReportLab)
# -----------------------------
REPORTLAB_OK = True
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
except Exception:
    REPORTLAB_OK = False


# -----------------------------
# Constantes / Guardrails
# -----------------------------
APP_TITLE = "ValuationSuite USIL (PYG)"
MIN_SPREAD = 0.005  # WACC debe ser > g∞ + 0.5%
COMMITTEE_MAX_PROB_NEG = 0.20  # 20%
PYG_SYMBOL = "Gs."
DEFAULT_SIMS = 15000


# -----------------------------
# Helpers formato
# -----------------------------
def fmt_gs(x: float | int) -> str:
    try:
        x = float(x)
    except Exception:
        return f"{PYG_SYMBOL} —"
    s = f"{x:,.0f}".replace(",", ".")  # 1,234,567 -> 1.234.567
    return f"{PYG_SYMBOL} {s}"

def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)

def wrap_lines(s: str, width: int = 92) -> list[str]:
    return textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False)

def safe_irr(cashflows: list[float]) -> float | None:
    """TIR puede no existir o ser no-única; aquí devolvemos una estimación conservadora o None."""
    try:
        irr = float(npf.irr(cashflows))
        if math.isnan(irr) or math.isinf(irr):
            return None
        # filtro conservador
        if irr < -0.99 or irr > 3.0:
            return None
        return irr
    except Exception:
        return None

def payback_simple(cashflows: list[float]) -> float | None:
    """Payback simple (años). cashflows[0] suele ser negativo (CAPEX0)."""
    cum = 0.0
    for t, cf in enumerate(cashflows):
        cum += cf
        if t == 0:
            continue
        if cum >= 0:
            prev = cum - cf
            if cf == 0:
                return float(t)
            frac = (0 - prev) / cf
            return (t - 1) + frac
    return None

def payback_discounted(cashflows: list[float], rate: float) -> float | None:
    cum = 0.0
    for t, cf in enumerate(cashflows):
        pv = cf / ((1 + rate) ** t)
        cum += pv
        if t == 0:
            continue
        if cum >= 0:
            prev = cum - pv
            if pv == 0:
                return float(t)
            frac = (0 - prev) / pv
            return (t - 1) + frac
    return None


# -----------------------------
# Finanzas core (CAPM / WACC)
# -----------------------------
def levered_beta(beta_u: float, debt: float, equity: float, tax_rate: float) -> float:
    if equity <= 0:
        return float("nan")
    return beta_u * (1 + (1 - tax_rate) * (debt / equity))

def cost_of_equity(rf: float, erp: float, beta_l: float, crp: float) -> float:
    return rf + beta_l * erp + crp

def wacc(ke: float, kd: float, debt: float, equity: float, tax_rate: float) -> float:
    v = debt + equity
    if v <= 0:
        return float("nan")
    wd = debt / v
    we = equity / v
    return we * ke + wd * kd * (1 - tax_rate)


# -----------------------------
# Puente contable -> FCF (Año 1)
# -----------------------------
@dataclass
class BridgeYear1:
    sales: float
    var_costs: float
    fixed_costs: float
    depreciation: float
    capex1: float
    d_ar: float
    d_inv: float
    d_ap: float
    tax_rate: float

    def ebit(self) -> float:
        return self.sales - self.var_costs - self.fixed_costs - self.depreciation

    def nopat(self) -> float:
        return self.ebit() * (1 - self.tax_rate)

    def d_nwc(self) -> float:
        # ΔNWC = ΔAR + ΔINV − ΔAP
        return self.d_ar + self.d_inv - self.d_ap

    def fcf1(self) -> float:
        # FCF = NOPAT + Depreciación − CAPEX − ΔNWC
        return self.nopat() + self.depreciation - self.capex1 - self.d_nwc()


# -----------------------------
# DCF (determinístico)
# -----------------------------
@dataclass
class DCFResult:
    years: list[int]
    fcf: list[float]         # ciclo 1..N (sin TV)
    tv: float                # terminal value en N (no descontado)
    pv_fcf: float
    pv_tv: float
    npv: float
    cashflows_full: list[float]  # [ -capex0, fcf1..fcfN+tv ]
    irr: float | None
    pb: float | None
    pb_disc: float | None

def run_dcf(
    capex0: float,
    fcf1: float,
    n_years: int,
    g_exp: float,
    g_inf: float,
    discount_rate: float,
) -> DCFResult:
    years = list(range(1, n_years + 1))
    fcf = []
    for t in years:
        fcf_t = fcf1 * ((1 + g_exp) ** (t - 1))
        fcf.append(fcf_t)

    if discount_rate <= g_inf + MIN_SPREAD:
        # invalid: TV explota
        tv = float("nan")
        pv_tv = float("nan")
    else:
        tv = (fcf[-1] * (1 + g_inf)) / (discount_rate - g_inf)
        pv_tv = tv / ((1 + discount_rate) ** n_years)

    pv_fcf = sum(fcf[t - 1] / ((1 + discount_rate) ** t) for t in years)
    npv = (pv_fcf + pv_tv) - capex0

    cashflows_full = [-capex0] + fcf[:-1] + [fcf[-1] + (0.0 if math.isnan(tv) else tv)]
    irr = safe_irr(cashflows_full)
    pb = payback_simple(cashflows_full)
    pb_disc = payback_discounted(cashflows_full, discount_rate)

    return DCFResult(
        years=years,
        fcf=fcf,
        tv=tv,
        pv_fcf=pv_fcf,
        pv_tv=pv_tv,
        npv=npv,
        cashflows_full=cashflows_full,
        irr=irr,
        pb=pb,
        pb_disc=pb_disc,
    )


# -----------------------------
# Monte Carlo (triangular)
# -----------------------------
@dataclass
class MCResult:
    sims: int
    npv_samples: np.ndarray
    valid_share: float
    p5: float
    p50: float
    p95: float
    mean: float
    stdev: float
    cvar5: float
    prob_neg: float
    corr: dict[str, float]  # drivers

def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 100:
        return float("nan")
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

@st.cache_data(show_spinner=False)
def run_monte_carlo(
    sims: int,
    base_capex0: float,
    base_fcf1: float,
    n_years: int,
    g_inf: float,
    # Triangular inputs
    g_min: float, g_mode: float, g_max: float,
    wacc_min: float, wacc_mode: float, wacc_max: float,
    capex_min: float, capex_mode: float, capex_max: float,
    fcf_mult_min: float, fcf_mult_mode: float, fcf_mult_max: float,
) -> MCResult:
    rng = np.random.default_rng()  # sin semilla expuesta

    g_s = rng.triangular(g_min, g_mode, g_max, sims)
    w_s = rng.triangular(wacc_min, wacc_mode, wacc_max, sims)
    capex_s = rng.triangular(capex_min, capex_mode, capex_max, sims)
    mult_s = rng.triangular(fcf_mult_min, fcf_mult_mode, fcf_mult_max, sims)

    yrs = np.arange(1, n_years + 1)
    fcf1_s = base_fcf1 * mult_s
    fcf_paths = fcf1_s[:, None] * (1.0 + g_s)[:, None] ** (yrs[None, :] - 1)

    valid = w_s > (g_inf + MIN_SPREAD)
    idx = np.where(valid)[0]

    npv = np.full(sims, np.nan)
    if idx.size > 0:
        fcf_v = fcf_paths[idx, :]
        w_v = w_s[idx]
        capex_v = capex_s[idx]

        tv_v = (fcf_v[:, -1] * (1.0 + g_inf)) / (w_v - g_inf)
        fcf_v[:, -1] = fcf_v[:, -1] + tv_v

        disc = (1.0 + w_v)[:, None] ** (yrs[None, :])
        pv = np.sum(fcf_v / disc, axis=1)
        npv[idx] = pv - capex_v

    valid_share = float(np.isfinite(npv).mean())

    # percentiles sobre válidos
    finite = npv[np.isfinite(npv)]
    if finite.size == 0:
        p5 = p50 = p95 = mean = stdev = cvar5 = prob_neg = float("nan")
    else:
        p5, p50, p95 = np.percentile(finite, [5, 50, 95])
        mean = float(np.mean(finite))
        stdev = float(np.std(finite))
        prob_neg = float(np.mean(finite < 0))
        tail = finite[finite <= p5]
        cvar5 = float(np.mean(tail)) if tail.size else p5

    corr = {
        "g": safe_corr(g_s[idx], npv[idx]) if idx.size else float("nan"),
        "WACC": safe_corr(w_s[idx], npv[idx]) if idx.size else float("nan"),
        "CAPEX0": safe_corr(capex_s[idx], npv[idx]) if idx.size else float("nan"),
        "Shock FCF1": safe_corr(mult_s[idx], npv[idx]) if idx.size else float("nan"),
    }

    return MCResult(
        sims=sims,
        npv_samples=npv,
        valid_share=valid_share,
        p5=float(p5),
        p50=float(p50),
        p95=float(p95),
        mean=float(mean),
        stdev=float(stdev),
        cvar5=float(cvar5),
        prob_neg=float(prob_neg),
        corr=corr,
    )


# -----------------------------
# Comité (default hard-coded)
# -----------------------------
@dataclass
class CommitteeResult:
    verdict: str
    rationale: str
    checks: list[tuple[str, bool]]

def committee_eval(npv_base: float, mc: MCResult) -> CommitteeResult:
    checks = [
        ("VAN base > 0", npv_base > 0),
        (f"P(VAN<0) ≤ {int(COMMITTEE_MAX_PROB_NEG*100)}%", mc.prob_neg <= COMMITTEE_MAX_PROB_NEG),
        ("P50(VAN) > 0", mc.p50 > 0),
        ("P5(VAN) > 0", mc.p5 > 0),
    ]
    ok = sum(1 for _, v in checks if v)
    if ok == len(checks):
        verdict = "APROBADO"
        rationale = (
            "El proyecto cumple criterios conservadores: creación de valor y downside controlado "
            "bajo incertidumbre razonable."
        )
    elif ok == 0:
        verdict = "RECHAZADO"
        rationale = (
            "El proyecto no cumple criterios mínimos del Comité. Se requiere rediseño del caso "
            "y fortalecimiento de supuestos antes de considerar aprobación."
        )
    else:
        verdict = "OBSERVADO"
        rationale = (
            "El proyecto muestra potencial, pero no cumple todos los criterios conservadores. "
            "Recomendación: reforzar evidencia, mitigaciones y supuestos críticos antes de aprobación."
        )
    return CommitteeResult(verdict=verdict, rationale=rationale, checks=checks)

def verdict_badge(v: str) -> str:
    return {"APROBADO": "✅", "OBSERVADO": "⚠️", "RECHAZADO": "⛔"}.get(v, "—")


# -----------------------------
# PDF One-Pager (A4)
# -----------------------------
@dataclass
class OnePager:
    project: str
    responsible: str
    today: str
    # KPIs
    npv: float
    irr: float | None
    pb: float | None
    pb_disc: float | None
    # supuestos
    n_years: int
    g_exp: float
    g_inf: float
    wacc: float
    ke: float
    kd: float
    capex0: float
    fcf1: float
    # bridge
    ebit: float
    nopat: float
    d_nwc: float
    capex1: float
    # Monte Carlo
    sims: int
    prob_neg: float
    p5: float
    p50: float
    p95: float
    mean: float
    stdev: float
    cvar5: float
    valid_share: float
    # flows
    fcf: list[float]
    tv: float
    # comité
    verdict: str
    rationale: str
    checks: list[tuple[str, bool]]
    # drivers
    corr: dict[str, float]

def build_onepager_text(op: OnePager) -> str:
    irr_txt = fmt_pct(op.irr) if op.irr is not None else "N/A"
    pb_txt = f"{op.pb:.2f} años" if op.pb is not None else "N/A"
    pbd_txt = f"{op.pb_disc:.2f} años" if op.pb_disc is not None else "N/A"
    lines = []
    lines.append("ONE-PAGER EJECUTIVO — EVALUACIÓN FINANCIERA")
    lines.append("Universidad San Ignacio de Loyola (USIL) — MBA — Proyectos de Inversión / Valuation")
    lines.append("Moneda: PYG")
    lines.append(f"Fecha: {op.today}")
    lines.append("")
    lines.append(f"Proyecto: {op.project} | Responsable: {op.responsible}")
    lines.append(f"DICTAMEN: {op.verdict} {verdict_badge(op.verdict)}")
    lines.append(op.rationale)
    lines.append("")
    lines.append("KPIs (determinístico)")
    lines.append(f"- VAN (base): {fmt_gs(op.npv)}")
    lines.append(f"- TIR (base): {irr_txt}")
    lines.append(f"- Payback simple: {pb_txt}")
    lines.append(f"- Payback descontado: {pbd_txt}")
    lines.append("")
    lines.append("Flujos proyectados (FCF) — ciclo 1..N")
    for i, v in enumerate(op.fcf, start=1):
        lines.append(f"- Año {i}: {fmt_gs(v)}")
    lines.append(f"- Valor terminal (TV, en año {op.n_years}): {fmt_gs(op.tv)}")
    lines.append("")
    lines.append("Supuestos clave")
    lines.append(f"- Horizonte: {op.n_years} años + perpetuidad | g explícito {fmt_pct(op.g_exp)} | g∞ {fmt_pct(op.g_inf)}")
    lines.append(f"- WACC: {fmt_pct(op.wacc)} (Ke {fmt_pct(op.ke)} | Kd {fmt_pct(op.kd)})")
    lines.append(f"- CAPEX 0: {fmt_gs(op.capex0)} | FCF Año 1 (calculado): {fmt_gs(op.fcf1)}")
    lines.append("")
    lines.append("Puente contable → FCF (Año 1)")
    lines.append(f"- EBIT: {fmt_gs(op.ebit)} | NOPAT: {fmt_gs(op.nopat)}")
    lines.append(f"- ΔCT (AR+INV−AP): {fmt_gs(op.d_nwc)} | CAPEX Año 1: {fmt_gs(op.capex1)}")
    lines.append("")
    lines.append("Monte Carlo — resumen ejecutivo")
    lines.append(f"- Simulaciones: {op.sims:,} | válidas: {op.valid_share*100:.1f}% | P(VAN<0): {op.prob_neg*100:.1f}%")
    lines.append(f"- P5: {fmt_gs(op.p5)} | P50: {fmt_gs(op.p50)} | P95: {fmt_gs(op.p95)}")
    lines.append(f"- Media: {fmt_gs(op.mean)} | σ: {fmt_gs(op.stdev)} | CVaR5: {fmt_gs(op.cvar5)}")
    lines.append("")
    lines.append("Checklist Comité (automático)")
    for k, ok in op.checks:
        lines.append(f"- {'✅' if ok else '❌'} {k}")
    lines.append("")
    lines.append("Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence.")
    return "\n".join(lines)

def generate_pdf_onepager(op: OnePager) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab no está disponible. Agrega `reportlab` a requirements.txt.")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    margin = 16 * mm
    x0 = margin
    y = h - margin

    def draw_h1(txt):
        nonlocal y
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x0, y, txt)
        y -= 8 * mm

    def draw_h2(txt):
        nonlocal y
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(x0, y, txt)
        y -= 6 * mm

    def draw_p(txt, size=9.5, width=95):
        nonlocal y
        c.setFont("Helvetica", size)
        for ln in wrap_lines(txt, width=width):
            c.drawString(x0, y, ln)
            y -= 4.6 * mm

    def draw_kv(k, v, colx):
        nonlocal y
        c.setFont("Helvetica-Bold", 9.2)
        c.drawString(colx, y, k)
        c.setFont("Helvetica", 9.2)
        c.drawRightString(colx + 70*mm, y, v)
        y -= 4.8 * mm

    # Header
    draw_h1("ONE-PAGER EJECUTIVO — EVALUACIÓN FINANCIERA (PYG)")
    draw_p(f"Universidad San Ignacio de Loyola (USIL) — MBA — Proyectos de Inversión / Valuation   |   Fecha: {op.today}", size=9.3, width=120)
    draw_p(f"Proyecto: {op.project}   |   Responsable: {op.responsible}", size=9.3, width=120)

    # Verdict banner
    y -= 2 * mm
    c.setFillColorRGB(0.08, 0.35, 0.18) if op.verdict == "APROBADO" else c.setFillColorRGB(0.45, 0.32, 0.08) if op.verdict == "OBSERVADO" else c.setFillColorRGB(0.45, 0.10, 0.10)
    c.roundRect(x0, y-10*mm, w-2*margin, 10*mm, 4*mm, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x0 + 4*mm, y-6.8*mm, f"DICTAMEN: {op.verdict} {verdict_badge(op.verdict)}")
    c.setFont("Helvetica", 9.2)
    c.drawRightString(w - margin - 4*mm, y-6.8*mm, "Checklist automático + Monte Carlo")
    c.setFillColorRGB(0, 0, 0)
    y -= 14 * mm

    # 2-column layout
    col1 = x0
    col2 = x0 + (w - 2*margin) / 2 + 4*mm
    colw = (w - 2*margin) / 2 - 4*mm

    # KPIs left
    y_left = y
    c.setFont("Helvetica-Bold", 10.5); c.drawString(col1, y_left, "KPIs (determinístico)")
    y_left -= 6 * mm
    for k, v in [
        ("VAN (base)", fmt_gs(op.npv)),
        ("TIR (base)", fmt_pct(op.irr) if op.irr is not None else "N/A"),
        ("Payback simple", f"{op.pb:.2f} años" if op.pb is not None else "N/A"),
        ("Payback descont.", f"{op.pb_disc:.2f} años" if op.pb_disc is not None else "N/A"),
    ]:
        c.setFont("Helvetica", 9.4)
        c.drawString(col1, y_left, f"• {k}: {v}")
        y_left -= 5 * mm

    y_left -= 2*mm
    c.setFont("Helvetica-Bold", 10.5); c.drawString(col1, y_left, "Supuestos clave")
    y_left -= 6*mm
    for ln in [
        f"Horizonte: {op.n_years} años + perpetuidad",
        f"g explícito: {fmt_pct(op.g_exp)} | g∞: {fmt_pct(op.g_inf)}",
        f"WACC: {fmt_pct(op.wacc)} (Ke {fmt_pct(op.ke)} | Kd {fmt_pct(op.kd)})",
        f"CAPEX0: {fmt_gs(op.capex0)} | FCF1: {fmt_gs(op.fcf1)}",
    ]:
        c.setFont("Helvetica", 9.4)
        c.drawString(col1, y_left, f"• {ln}")
        y_left -= 5*mm

    # Monte Carlo right
    y_right = y
    c.setFont("Helvetica-Bold", 10.5); c.drawString(col2, y_right, "Monte Carlo — lectura ejecutiva")
    y_right -= 6*mm
    for ln in [
        f"Simulaciones: {op.sims:,} | válidas: {op.valid_share*100:.1f}% | P(VAN<0): {op.prob_neg*100:.1f}%",
        f"P5/P50/P95: {fmt_gs(op.p5)} / {fmt_gs(op.p50)} / {fmt_gs(op.p95)}",
        f"Media: {fmt_gs(op.mean)} | σ: {fmt_gs(op.stdev)} | CVaR5: {fmt_gs(op.cvar5)}",
    ]:
        c.setFont("Helvetica", 9.2)
        for ln2 in wrap_lines(ln, width=62):
            c.drawString(col2, y_right, ln2)
            y_right -= 4.6*mm

    # tiny histogram (bars) in PDF
    y_right -= 2*mm
    finite = op.p50  # placeholder for scale
    # compute histogram from stored samples? not available in op, so draw percentile marks as proxy
    # We'll draw a simple axis with P5/P50/P95 markers.
    axis_y = y_right - 6*mm
    axis_x0 = col2
    axis_x1 = col2 + colw
    c.setLineWidth(0.8)
    c.line(axis_x0, axis_y, axis_x1, axis_y)
    # map percentiles to [0,1] using p5..p95
    pmin = op.p5
    pmax = op.p95
    def xmap(v):
        if not (pmax > pmin):
            return axis_x0 + colw/2
        return axis_x0 + (v - pmin) / (pmax - pmin) * colw

    for lab, val in [("P5", op.p5), ("P50", op.p50), ("P95", op.p95)]:
        x = xmap(val)
        c.setLineWidth(1.2)
        c.line(x, axis_y-4*mm, x, axis_y+4*mm)
        c.setFont("Helvetica", 8.6)
        c.drawCentredString(x, axis_y+5.2*mm, lab)
    y_right = axis_y - 10*mm

    # Flujos bottom (full width)
    y_bottom = min(y_left, y_right) - 3*mm
    y = y_bottom
    draw_h2("Flujos proyectados (FCF) — ciclo 1..N + TV (TV separado)")
    # list 1..N
    c.setFont("Helvetica", 9.2)
    for i, v in enumerate(op.fcf, start=1):
        c.drawString(x0, y, f"• Año {i}: {fmt_gs(v)}")
        y -= 4.6*mm
        if y < 28*mm:
            break
    c.drawString(x0, y, f"• TV (en año {op.n_years}): {fmt_gs(op.tv)}")
    y -= 6*mm

    draw_h2("Checklist Comité (automático)")
    c.setFont("Helvetica", 9.2)
    for k, ok in op.checks:
        c.drawString(x0, y, f"• {'✅' if ok else '❌'} {k}")
        y -= 4.6*mm

    # Footer
    c.setFont("Helvetica-Oblique", 8.6)
    c.drawString(x0, 12*mm, "Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence.")
    c.drawRightString(w - margin, 12*mm, "ValuationSuite USIL (PYG)")

    c.showPage()
    c.save()
    return buf.getvalue()


# -----------------------------
# Streamlit UI (premium-ish)
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

# CSS: tarjetas + spacing (evita números “pegados”)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
      div[data-testid="stMetric"] { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07);
        padding: 14px 14px 10px 14px; border-radius: 14px; }
      div[data-testid="stMetric"] label { font-size: 0.85rem; opacity: 0.9; }
      div[data-testid="stMetric"] div { gap: 8px; }
      .usil-card { background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 16px 16px 10px 16px; }
      .usil-title { font-size: 1.25rem; font-weight: 700; margin: 0 0 6px 0; }
      .usil-sub { opacity: 0.9; margin: 0 0 10px 0; }
      .small-muted { opacity: 0.75; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 ValuationSuite USIL — Evaluación de Proyectos (PYG)")
st.caption("DCF + Monte Carlo + Comité (default) + One-Pager ejecutivo (PDF opcional).")

# -----------------------------
# Sidebar inputs (estructura pedagógica)
# -----------------------------
st.sidebar.header("🧩 Identificación")
project = st.sidebar.text_input("Proyecto", value="Proyecto")
responsible = st.sidebar.text_input("Responsable", value="Docente: Jorge Rojas")

st.sidebar.divider()
st.sidebar.header("0) Inversión inicial")
capex0 = st.sidebar.number_input("CAPEX Año 0 (inversión inicial)", min_value=1.0, value=3_500_000_000.0, step=50_000_000.0)

st.sidebar.divider()
st.sidebar.header("1) CAPM / WACC (Nominal, D/E contable)")
rf = st.sidebar.number_input("Rf (%)", value=4.50, step=0.10) / 100
erp = st.sidebar.number_input("ERP (%)", value=5.50, step=0.10) / 100
crp = st.sidebar.number_input("CRP (%)", value=2.00, step=0.10) / 100
beta_u = st.sidebar.number_input("βU (desapalancada)", value=0.90, step=0.05)
tax_rate = st.sidebar.number_input("Impuesto (T) (%)", value=10.0, step=0.5) / 100

st.sidebar.divider()
st.sidebar.header("2) Estructura de capital (valores contables)")
debt = st.sidebar.number_input("Deuda (D)", value=4_000_000_000.0, step=50_000_000.0, min_value=0.0)
equity = st.sidebar.number_input("Capital propio (E)", value=6_000_000_000.0, step=50_000_000.0, min_value=1.0)
kd = st.sidebar.number_input("Costo de deuda Kd (%)", value=7.00, step=0.10) / 100

# CAPM/WACC
beta_l = levered_beta(beta_u, debt, equity, tax_rate)
ke = cost_of_equity(rf, erp, beta_l, crp)
wacc_rate = wacc(ke, kd, debt, equity, tax_rate)

st.sidebar.divider()
st.sidebar.header("3A) Contable Año 1 (para calcular FCF Año 1)")
sales = st.sidebar.number_input("Ventas Año 1", value=6_000_000_000.0, step=50_000_000.0, min_value=0.0)
var_costs = st.sidebar.number_input("Costos variables Año 1", value=3_000_000_000.0, step=50_000_000.0, min_value=0.0)
fixed_costs = st.sidebar.number_input("Costos fijos Año 1", value=1_500_000_000.0, step=50_000_000.0, min_value=0.0)
depr = st.sidebar.number_input("Depreciación Año 1 (no caja)", value=250_000_000.0, step=10_000_000.0, min_value=0.0)
capex1 = st.sidebar.number_input("CAPEX Año 1 (mantenimiento/crecimiento)", value=250_000_000.0, step=10_000_000.0, min_value=0.0)

st.sidebar.caption("🧠 Regla pedagógica: EBIT = Ventas − CV − CF − Depreciación.")

st.sidebar.subheader("Δ Capital de trabajo (Año 1) — componentes")
d_ar = st.sidebar.number_input("Δ Cuentas por cobrar (AR)", value=100_000_000.0, step=5_000_000.0)
d_inv = st.sidebar.number_input("Δ Inventarios (INV)", value=80_000_000.0, step=5_000_000.0)
d_ap = st.sidebar.number_input("Δ Cuentas por pagar (AP)", value=40_000_000.0, step=5_000_000.0)

bridge = BridgeYear1(
    sales=sales,
    var_costs=var_costs,
    fixed_costs=fixed_costs,
    depreciation=depr,
    capex1=capex1,
    d_ar=d_ar,
    d_inv=d_inv,
    d_ap=d_ap,
    tax_rate=tax_rate,
)

fcf1 = bridge.fcf1()

st.sidebar.divider()
st.sidebar.header("3B) Flujos (Ciclo 1..N) + g perpetuidad")
n_years = int(st.sidebar.slider("Años de proyección (N)", min_value=3, max_value=10, value=5))
g_exp = st.sidebar.number_input("g explícito (%) (crecimiento ciclo 1..N)", value=5.00, step=0.25) / 100
g_inf = st.sidebar.number_input("g a perpetuidad (%)", value=2.00, step=0.10) / 100

st.sidebar.divider()
st.sidebar.header("4) Monte Carlo (incertidumbre razonable)")
sims = int(st.sidebar.slider("Simulaciones", min_value=5_000, max_value=60_000, value=DEFAULT_SIMS, step=1_000))
st.sidebar.caption("Se asume RNG interno (sin semilla expuesta).")

st.sidebar.subheader("Rangos triangulares (mín = adverso, base = más probable, máx = favorable)")
g_min = st.sidebar.number_input("g mín (%)", value=1.00, step=0.25) / 100
g_mode = st.sidebar.number_input("g base (%)", value=5.00, step=0.25) / 100
g_max = st.sidebar.number_input("g máx (%)", value=9.00, step=0.25) / 100

wacc_band = st.sidebar.number_input("WACC rango ± (%)", value=2.00, step=0.25) / 100
w_min, w_mode, w_max = max(0.0001, wacc_rate - wacc_band), wacc_rate, wacc_rate + wacc_band

capex_min = st.sidebar.number_input("CAPEX mín (favorable)", value=capex0 * 0.90, step=50_000_000.0)
capex_mode = st.sidebar.number_input("CAPEX base", value=capex0, step=50_000_000.0)
capex_max = st.sidebar.number_input("CAPEX máx (adverso)", value=capex0 * 1.10, step=50_000_000.0)

mult_min = st.sidebar.number_input("Shock FCF1 mín (adverso)", value=0.85, step=0.01)
mult_mode = st.sidebar.number_input("Shock FCF1 base", value=1.00, step=0.01)
mult_max = st.sidebar.number_input("Shock FCF1 máx (favorable)", value=1.15, step=0.01)


# -----------------------------
# Validaciones rápidas
# -----------------------------
if not (g_min <= g_mode <= g_max):
    st.sidebar.error("Rango g triangular inválido: debe cumplirse g mín ≤ g base ≤ g máx.")
if not (capex_min <= capex_mode <= capex_max):
    st.sidebar.error("Rango CAPEX triangular inválido: debe cumplirse mín ≤ base ≤ máx.")
if not (mult_min <= mult_mode <= mult_max):
    st.sidebar.error("Rango shock triangular inválido: debe cumplirse mín ≤ base ≤ máx.")


# -----------------------------
# Cálculos
# -----------------------------
dcf = run_dcf(capex0=capex0, fcf1=fcf1, n_years=n_years, g_exp=g_exp, g_inf=g_inf, discount_rate=wacc_rate)

mc = run_monte_carlo(
    sims=sims,
    base_capex0=capex0,
    base_fcf1=fcf1,
    n_years=n_years,
    g_inf=g_inf,
    g_min=g_min, g_mode=g_mode, g_max=g_max,
    wacc_min=w_min, wacc_mode=w_mode, wacc_max=w_max,
    capex_min=capex_min, capex_mode=capex_mode, capex_max=capex_max,
    fcf_mult_min=mult_min, fcf_mult_mode=mult_mode, fcf_mult_max=mult_max
)

committee = committee_eval(dcf.npv, mc)

# -----------------------------
# Dashboard (one page)
# -----------------------------
# Row 1: KPIs
k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
k1.metric("VAN (base)", fmt_gs(dcf.npv), help="Determinístico: PV(FCF ciclo 1..N) + PV(TV) − CAPEX0")
k2.metric("TIR (base)", fmt_pct(dcf.irr) if dcf.irr is not None else "N/A", help="Determinístico (puede no existir / no-única)")
k3.metric("Payback", f"{dcf.pb:.2f} años" if dcf.pb is not None else "N/A", help="Simple (sin descuento)")
k4.metric("Payback (desc.)", f"{dcf.pb_disc:.2f} años" if dcf.pb_disc is not None else "N/A", help="Descontado al WACC")
k5.metric("P(VAN<0)", f"{mc.prob_neg*100:.1f}%", help="Monte Carlo sobre simulaciones válidas")
k6.metric("P50 (VAN)", fmt_gs(mc.p50), help="Percentil 50 de la distribución del VAN (Monte Carlo)")

# Row 2: dictamen + estructura valor
c1, c2 = st.columns([1.15, 1.0], gap="large")

with c1:
    st.markdown('<div class="usil-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="usil-title">Decisión & supuestos</div>', unsafe_allow_html=True)
    st.markdown(
        f"<b>DICTAMEN:</b> {committee.verdict} {verdict_badge(committee.verdict)}<br>"
        f"<span class='small-muted'>{committee.rationale}</span>",
        unsafe_allow_html=True
    )

    st.markdown("**Supuestos clave**")
    st.write(
        f"- Horizonte: **{n_years} años + perpetuidad**\n"
        f"- g explícito: **{fmt_pct(g_exp)}** | g∞: **{fmt_pct(g_inf)}**\n"
        f"- WACC: **{fmt_pct(wacc_rate)}** (Ke {fmt_pct(ke)} | Kd {fmt_pct(kd)})\n"
        f"- CAPEX0: **{fmt_gs(capex0)}** | FCF Año 1 (calculado): **{fmt_gs(fcf1)}**"
    )

    st.markdown("**Checklist Comité (automático)**")
    for label, ok in committee.checks:
        st.write(f"{'✅' if ok else '❌'} {label}")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="usil-card">', unsafe_allow_html=True)
    st.markdown('<div class="usil-title">Estructura del valor</div>', unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>VAN = PV(FCF ciclo 1..N) + PV(TV) − CAPEX₀. TV se reporta separado para transparencia.</div>", unsafe_allow_html=True)

    # Stacked bar: -CAPEX, PV FCF, PV TV
    fig_value = go.Figure()
    fig_value.add_trace(go.Bar(name="PV FCF", x=["VAN"], y=[dcf.pv_fcf]))
    fig_value.add_trace(go.Bar(name="PV TV", x=["VAN"], y=[dcf.pv_tv]))
    fig_value.add_trace(go.Bar(name="CAPEX₀", x=["VAN"], y=[-capex0]))
    fig_value.update_layout(barmode="relative", height=260, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    fig_value.update_yaxes(title_text="Gs.")
    st.plotly_chart(fig_value, use_container_width=True)

    st.success(f"**VAN (base): {fmt_gs(dcf.npv)}**")
    st.markdown('</div>', unsafe_allow_html=True)

# Row 3: Flujos + Riesgo
c3, c4 = st.columns([1.05, 1.0], gap="large")

with c3:
    st.markdown('<div class="usil-card">', unsafe_allow_html=True)
    st.markdown('<div class="usil-title">Flujos proyectados (ciclo 1..N) + TV</div>', unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>FCF por ciclo. El valor terminal (TV) se muestra separado.</div>", unsafe_allow_html=True)

    years = [f"{t}" for t in dcf.years]
    fig_fcf = go.Figure()
    fig_fcf.add_trace(go.Bar(x=years, y=dcf.fcf, name="FCF"))
    fig_fcf.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Año", yaxis_title="Gs.")
    st.plotly_chart(fig_fcf, use_container_width=True)
    st.caption(f"**TV (en ciclo {n_years}): {fmt_gs(dcf.tv)}**")
    st.markdown('</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="usil-card">', unsafe_allow_html=True)
    st.markdown('<div class="usil-title">Monte Carlo — distribución del VAN</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='small-muted'>Percentiles (P5/P50/P95) + métricas de riesgo (σ, CVaR5) "
        "para lectura ejecutiva.</div>",
        unsafe_allow_html=True
    )

    finite = mc.npv_samples[np.isfinite(mc.npv_samples)]
    if finite.size == 0:
        st.error("No hay simulaciones válidas (revisar WACC vs g∞).")
    else:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=finite, nbinsx=40, name="Distribución VAN"))
        for val, name in [(mc.p5, "P5"), (mc.p50, "P50"), (mc.p95, "P95")]:
            fig_hist.add_vline(x=val, line_dash="dash", annotation_text=name, annotation_position="top")
        fig_hist.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="VAN (Gs.)", yaxis_title="count")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.write(
            f"**P5/P50/P95:** {fmt_gs(mc.p5)} / {fmt_gs(mc.p50)} / {fmt_gs(mc.p95)}  \n"
            f"**Media:** {fmt_gs(mc.mean)} · **σ:** {fmt_gs(mc.stdev)} · **CVaR5:** {fmt_gs(mc.cvar5)}  \n"
            f"**P(VAN<0):** {mc.prob_neg*100:.1f}% · **Simulaciones:** {mc.sims:,} · **Válidas:** {mc.valid_share*100:.1f}%"
        )

        # Drivers (correlación) — mini gráfico
        corr_items = [(k, v) for k, v in mc.corr.items()]
        corr_x = [k for k, _ in corr_items]
        corr_y = [0 if (v is None or math.isnan(v)) else v for _, v in corr_items]
        fig_drv = go.Figure()
        fig_drv.add_trace(go.Bar(x=corr_x, y=corr_y, name="Correlación con VAN"))
        fig_drv.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="corr")
        st.plotly_chart(fig_drv, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Bridge card (extra row) - pedagógico
st.markdown("### Puente contable → FCF (Año 1) (pedagógico)")
b1, b2, b3, b4 = st.columns(4, gap="small")
b1.metric("EBIT", fmt_gs(bridge.ebit()))
b2.metric("NOPAT", fmt_gs(bridge.nopat()))
b3.metric("ΔCT (AR+INV−AP)", fmt_gs(bridge.d_nwc()))
b4.metric("FCF Año 1 (calculado)", fmt_gs(fcf1))

st.caption("FCF = NOPAT + Depreciación − CAPEX − ΔNWC. ΔNWC = ΔAR + ΔINV − ΔAP.")

st.divider()

# Export
st.header("Export")
colx, coly = st.columns([1, 1], gap="large")

# Build onepager object
op = OnePager(
    project=project,
    responsible=responsible,
    today=date.today().isoformat(),
    npv=dcf.npv,
    irr=dcf.irr,
    pb=dcf.pb,
    pb_disc=dcf.pb_disc,
    n_years=n_years,
    g_exp=g_exp,
    g_inf=g_inf,
    wacc=wacc_rate,
    ke=ke,
    kd=kd,
    capex0=capex0,
    fcf1=fcf1,
    ebit=bridge.ebit(),
    nopat=bridge.nopat(),
    d_nwc=bridge.d_nwc(),
    capex1=capex1,
    sims=mc.sims,
    prob_neg=mc.prob_neg,
    p5=mc.p5,
    p50=mc.p50,
    p95=mc.p95,
    mean=mc.mean,
    stdev=mc.stdev,
    cvar5=mc.cvar5,
    valid_share=mc.valid_share,
    fcf=dcf.fcf,
    tv=dcf.tv,
    verdict=committee.verdict,
    rationale=committee.rationale,
    checks=committee.checks,
    corr=mc.corr,
)

onepager_txt = build_onepager_text(op)

with colx:
    st.download_button(
        "⬇️ Descargar One-Pager (TXT)",
        data=onepager_txt.encode("utf-8"),
        file_name="one_pager_usil_pyg.txt",
        mime="text/plain",
        use_container_width=True,
    )

with coly:
    if REPORTLAB_OK:
        try:
            pdf_bytes = generate_pdf_onepager(op)
            st.download_button(
                "⬇️ Descargar One-Pager (PDF)",
                data=pdf_bytes,
                file_name="one_pager_usil_pyg.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"No se pudo generar PDF: {e}")
    else:
        st.info("Para exportar PDF, agrega `reportlab` a requirements.txt (y redeploy).")

st.caption("Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence. · ValuationSuite USIL (PYG)")
