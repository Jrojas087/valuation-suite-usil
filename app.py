import streamlit as st
import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.express as px
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
    from reportlab.lib import colors
    from reportlab.lib.utils import ImageReader
except Exception:
    REPORTLAB_OK = False

# ============================================================
# Parámetros fijos (Paraguay / Comité)
# ============================================================
CURRENCY = "PYG"          # FIJO
MC_SEED = 42              # fijo interno (no se expone)
MIN_SPREAD = 0.005        # para TV en Monte Carlo: WACC > g∞ + 0.5%
COMMITTEE_MAX_PROB_NEG = 0.20  # P(VAN<0) ≤ 20%
# Comité fijo: P50>0 y P5>0
# ============================================================

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="ValuationApp – MBA USIL (PY)", layout="wide")
st.title("📊 ValuationApp – Evaluación Financiera de Proyectos (MBA – USIL)")
st.caption(
    "Marco: DCF (caso base) + lectura probabilística (Monte Carlo). "
    "Moneda fija: Guaraníes (PYG). Reporte: One-Pager ejecutivo estilo dashboard."
)

# ============================================================
# Helpers
# ============================================================
def pct(x: float) -> str:
    return f"{x:.2%}"

def money_pyg(x: float) -> str:
    # Formato Guaraní: Gs. 1.234.567.890
    return f"Gs. {x:,.0f}".replace(",", ".")

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
    # filtro conservador (evita valores absurdos por ruido numérico)
    if x < -0.99 or x > 2.0:
        return None
    return x

def detect_non_conventional_flows(cashflows) -> bool:
    signs = []
    for v in cashflows:
        if abs(v) < 1e-12:
            continue
        signs.append(1 if v > 0 else -1)
    if len(signs) < 2:
        return False
    changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    return changes > 1

def compute_discounted_payback(capex0: float, flows: np.ndarray, wacc: float):
    """
    Payback descontado: primer ciclo en que el acumulado PV(FCF) >= CAPEX.
    Retorna (payback_year:int|None, approx_years:float|None)
    """
    cum = -capex0
    for i, f in enumerate(flows, start=1):
        pv = f / ((1.0 + wacc) ** i)
        prev = cum
        cum += pv
        if cum >= 0:
            if pv == 0:
                return i, None
            frac = (0 - prev) / pv
            return i, float((i - 1) + frac)
    return None, None

def safe_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 50:
        return np.nan
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def committee_verdict(prob_neg: float, p50: float, p5: float):
    checks = [
        ("P(VAN<0) ≤ 20%", prob_neg <= COMMITTEE_MAX_PROB_NEG),
        ("P50(VAN) > 0", p50 > 0),
        ("P5(VAN) > 0", p5 > 0),
    ]
    ok = sum(v for _, v in checks)
    if ok == len(checks):
        return ("APROBADO",
                "El proyecto satisface criterios conservadores: creación de valor y downside controlado bajo incertidumbre razonable.")
    if ok == 0:
        return ("RECHAZADO",
                "El proyecto no cumple criterios mínimos: destrucción de valor y/o perfil de riesgo incompatible con una decisión favorable.")
    return ("OBSERVADO",
            "El proyecto presenta potencial, pero evidencia debilidades relevantes en robustez y/o downside. Requiere ajustes y mitigaciones previas.")

def recommended_actions(prob_neg, p5, p50, driver_focus: str | None):
    actions = []
    if (prob_neg <= COMMITTEE_MAX_PROB_NEG) and (p50 > 0) and (p5 > 0):
        actions.append("Avanzar a la siguiente etapa con control formativo: trazabilidad de supuestos y evidencia verificable.")
        actions.append("Formalizar mitigaciones (faseo, contratos, contingencias operativas) antes de comprometer capital incremental.")
        if driver_focus:
            actions.append(f"Priorizar validación y mitigación sobre el supuesto más sensible: {driver_focus}.")
        return actions

    actions.append("Replantear supuestos críticos y reforzar evidencia antes de una aprobación.")
    if prob_neg > COMMITTEE_MAX_PROB_NEG:
        actions.append("Reducir incertidumbre o rediseñar el caso: P(VAN<0) excede umbral de comité.")
    if p50 <= 0:
        actions.append("Mejorar robustez del caso base: revisar ingresos/costos, cronograma, CAPEX y estrategia comercial.")
    if p5 <= 0:
        actions.append("Mejorar protección del downside: faseo de inversión, ajustes de pricing/margen y gestión del capital de trabajo.")
    if driver_focus:
        actions.append(f"Variable prioritaria para intervención: {driver_focus}.")
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
    capex_min: float, capex_mode: float, capex_max: float,
    fcf_mult_min: float, fcf_mult_mode: float, fcf_mult_max: float,
    seed: int = MC_SEED
):
    rng = np.random.default_rng(seed)

    g_s = rng.triangular(g_min, g_mode, g_max, sims)
    w_s = rng.triangular(w_min, w_mode, w_max, sims)
    capex_s = rng.triangular(capex_min, capex_mode, capex_max, sims)
    mult_s = rng.triangular(fcf_mult_min, fcf_mult_mode, fcf_mult_max, sims)

    yrs = np.arange(1, n_years + 1)
    fcf1_s = fcf_y1 * mult_s
    fcf_paths = fcf1_s[:, None] * (1.0 + g_s)[:, None] ** (yrs[None, :] - 1)

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
# PDF – helpers
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

def _wrap(text: str, max_chars: int = 78):
    words = (text or "").split()
    lines, line = [], ""
    for w in words:
        if len(line) + len(w) + 1 <= max_chars:
            line = (line + " " + w).strip()
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines

def generate_onepager_dashboard_pdf(
    institution: str,
    program: str,
    course: str,
    project: str,
    responsible: str,
    n_years: int,
    wacc: float,
    g_exp: float,
    g_inf: float,
    capex0: float,
    fcf_path: np.ndarray,
    tv: float,
    van_base: float,
    irr_base: float | None,
    payback_text: str,
    sims: int,
    prob_neg: float,
    p5: float,
    p50: float,
    p95: float,
    verdict: str,
    rationale: str,
    driver_focus: str | None,
    mc_samples: np.ndarray | None,
    nonconv_warning: bool,
    logo_reader=None,
) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab no está disponible. Agregue `reportlab` a requirements.txt.")

    # Premium palette
    BG = colors.HexColor("#0B1220")
    CARD = colors.HexColor("#111B2E")
    CARD2 = colors.HexColor("#0F172A")
    TEXT = colors.HexColor("#E5E7EB")
    MUTED = colors.HexColor("#9CA3AF")
    BORDER = colors.HexColor("#1F2A44")

    ACCENT = colors.HexColor("#60A5FA")
    GOOD = colors.HexColor("#34D399")
    WARN = colors.HexColor("#FBBF24")
    BAD = colors.HexColor("#F87171")

    def vcolor(v: str):
        v = (v or "").upper()
        if v == "APROBADO":
            return GOOD
        if v == "OBSERVADO":
            return WARN
        if v == "RECHAZADO":
            return BAD
        return ACCENT

    def rr(c, x, y, w, h, r=14, fill_color=CARD, stroke_color=BORDER, stroke=0):
        c.setFillColor(fill_color)
        c.setStrokeColor(stroke_color)
        c.roundRect(x, y, w, h, r, fill=1, stroke=stroke)

    def text(c, x, y, s, size=10, bold=False, col=TEXT):
        c.setFillColor(col)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x, y, s)

    def small(c, x, y, s, size=9, col=MUTED):
        c.setFillColor(col)
        c.setFont("Helvetica", size)
        c.drawString(x, y, s)

    # waterfall components
    years = np.arange(1, n_years + 1)
    pv_fcf = float(np.sum(fcf_path / ((1.0 + wacc) ** years)))
    pv_tv = float(tv / ((1.0 + wacc) ** n_years))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter

    # Background
    c.setFillColor(BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    margin = 0.58 * inch

    # Header
    header_h = 0.85 * inch
    rr(c, margin, H - margin - header_h, W - 2 * margin, header_h, r=16, fill_color=CARD2)

    hx = margin + 0.22 * inch
    hy = H - margin - 0.22 * inch

    if logo_reader is not None:
        try:
            c.drawImage(logo_reader, hx, hy - 0.52 * inch, width=1.2 * inch, height=0.45 * inch, mask="auto")
            hx += 1.35 * inch
        except Exception:
            pass

    text(c, hx, hy - 0.26 * inch, "ONE-PAGER EJECUTIVO – EVALUACIÓN FINANCIERA", size=13, bold=True, col=TEXT)
    small(c, hx, hy - 0.50 * inch, f"{institution} · {program} · {course}", size=9.2, col=MUTED)

    small(c, W - margin - 2.3 * inch, hy - 0.26 * inch, f"Fecha: {date.today().isoformat()}", size=9.2, col=MUTED)
    small(c, W - margin - 2.3 * inch, hy - 0.50 * inch, f"Moneda: PYG (Gs.)", size=9.2, col=MUTED)

    # KPI strip
    kpi_y = H - margin - header_h - 0.18 * inch - 0.86 * inch
    kpi_h = 0.86 * inch
    gap = 0.16 * inch
    kpi_w = (W - 2 * margin - 4 * gap) / 5

    def kpi_card(x, y, title, value, subtitle, accent=ACCENT):
        rr(c, x, y, kpi_w, kpi_h, r=16, fill_color=CARD)
        c.setFillColor(accent)
        c.rect(x + 14, y + kpi_h - 12, kpi_w - 28, 2.4, fill=1, stroke=0)
        small(c, x + 14, y + kpi_h - 28, title, size=9.0, col=MUTED)
        text(c, x + 14, y + kpi_h - 52, value, size=14.2, bold=True, col=TEXT)
        small(c, x + 14, y + 14, subtitle, size=8.8, col=MUTED)

    irr_txt = pct(irr_base) if irr_base is not None else "N/A"
    x0 = margin
    kpi_card(x0, kpi_y, "VAN (caso base)", money_pyg(van_base), "Número principal", ACCENT); x0 += kpi_w + gap
    kpi_card(x0, kpi_y, "TIR (caso base)", irr_txt, f"vs WACC {pct(wacc)}", ACCENT); x0 += kpi_w + gap
    kpi_card(x0, kpi_y, "Payback (descont.)", payback_text, "Recuperación económica", ACCENT); x0 += kpi_w + gap

    # KPI 4: risk
    kpi_card(x0, kpi_y, "P(VAN<0)", f"{prob_neg:.1%}", "Riesgo downside", vcolor(verdict)); x0 += kpi_w + gap
    kpi_card(x0, kpi_y, "P5 (VAN)", money_pyg(p5), "Downside plausible", vcolor(verdict))

    # Grid
    col_w = (W - 2 * margin - gap) / 2
    left_x = margin
    right_x = margin + col_w + gap

    top_pan_h = 2.12 * inch
    top_pan_y = kpi_y - 0.18 * inch - top_pan_h

    # Panel: Decision (left)
    rr(c, left_x, top_pan_y, col_w, top_pan_h, r=16, fill_color=CARD)
    text(c, left_x + 14, top_pan_y + top_pan_h - 22, "Decisión & supuestos", size=10.8, bold=True)
    c.setStrokeColor(BORDER); c.setLineWidth(1)
    c.line(left_x + 14, top_pan_y + top_pan_h - 30, left_x + col_w - 14, top_pan_y + top_pan_h - 30)

    yy = top_pan_y + top_pan_h - 50
    text(c, left_x + 14, yy, f"Proyecto: {project[:58] + ('…' if len(project) > 58 else '')}", size=9.6, bold=True); yy -= 14
    small(c, left_x + 14, yy, f"Horizonte: {n_years} años · g: {pct(g_exp)} · g∞: {pct(g_inf)}", size=9.2); yy -= 13
    small(c, left_x + 14, yy, f"WACC: {pct(wacc)} · CAPEX: {money_pyg(capex0)} · FCF Año 1: {money_pyg(float(fcf_path[0]))}", size=9.2); yy -= 16

    text(c, left_x + 14, yy, f"DICTAMEN: {verdict}", size=12, bold=True, col=vcolor(verdict)); yy -= 14
    for line in _wrap(rationale, 80)[:3]:
        small(c, left_x + 14, yy, line, size=9.0); yy -= 12

    if driver_focus:
        small(c, left_x + 14, yy - 2, f"Sensibilidad orientativa: {driver_focus[:66] + ('…' if len(driver_focus) > 66 else '')}", size=9.0)
        yy -= 14

    if nonconv_warning:
        small(c, left_x + 14, top_pan_y + 14,
              "Nota: flujos no convencionales; la TIR puede no ser única. En comité se prioriza VAN + riesgo.",
              size=8.6, col=WARN)

    # Panel: Waterfall (right)
    rr(c, right_x, top_pan_y, col_w, top_pan_h, r=16, fill_color=CARD)
    text(c, right_x + 14, top_pan_y + top_pan_h - 22, "Estructura del valor", size=10.8, bold=True)
    c.line(right_x + 14, top_pan_y + top_pan_h - 30, right_x + col_w - 14, top_pan_y + top_pan_h - 30)

    small(c, right_x + 14, top_pan_y + top_pan_h - 46,
          "VAN = PV(FCF 1..N) + PV(TV) − CAPEX. TV se reporta separado para evitar ambigüedades.",
          size=9.0)

    wf_x = right_x + 14
    wf_y = top_pan_y + 44
    wf_w = col_w - 28
    wf_h = top_pan_h - 96
    rr(c, wf_x, wf_y, wf_w, wf_h, r=12, fill_color=colors.HexColor("#0D1528"), stroke=0)

    comps = [(-capex0, "CAPEX"), (pv_fcf, "PV FCF"), (pv_tv, "PV TV")]
    running = 0
    run_vals = []
    for val, _ in comps:
        running += val
        run_vals.append(running)
    lo = min([0] + run_vals)
    hi = max([0] + run_vals)
    span = (hi - lo) if (hi - lo) != 0 else 1.0

    def y_of(v):
        return wf_y + 10 + (v - lo) / span * (wf_h - 20)

    c.setStrokeColor(BORDER)
    c.setLineWidth(1)
    c.line(wf_x + 10, y_of(0), wf_x + wf_w - 10, y_of(0))

    bar_w = (wf_w - 30) / 4
    cur_x = wf_x + 12
    base = 0
    for val, label in comps:
        top = base + val
        y0 = y_of(min(base, top))
        yh = abs(y_of(top) - y_of(base))
        c.setFillColor(ACCENT if val >= 0 else BAD)
        c.rect(cur_x, y0, bar_w, yh, fill=1, stroke=0)
        c.setStrokeColor(BORDER)
        c.line(cur_x + bar_w, y_of(top), cur_x + bar_w + 6, y_of(top))
        c.setFillColor(MUTED); c.setFont("Helvetica", 8.2)
        c.drawString(cur_x, wf_y + 6, label)
        base = top
        cur_x += bar_w + 8

    c.setFillColor(vcolor(verdict))
    c.roundRect(wf_x + wf_w - 132, wf_y + wf_h - 28, 120, 18, 8, fill=1, stroke=0)
    c.setFillColor(BG); c.setFont("Helvetica-Bold", 9.4)
    c.drawString(wf_x + wf_w - 126, wf_y + wf_h - 24, f"VAN: {money_pyg(van_base)}")

    # Bottom panels
    bottom_h = 2.62 * inch
    bottom_y = margin + 0.1 * inch

    # Flows (left bottom)
    rr(c, left_x, bottom_y, col_w, bottom_h, r=16, fill_color=CARD)
    text(c, left_x + 14, bottom_y + bottom_h - 22, "Flujos proyectados (Ciclo 1…N) + TV", size=10.8, bold=True)
    c.line(left_x + 14, bottom_y + bottom_h - 30, left_x + col_w - 14, bottom_y + bottom_h - 30)

    bx = left_x + 14
    by = bottom_y + bottom_h - 30 - 1.25 * inch
    bw = col_w - 28
    bh = 1.12 * inch
    rr(c, bx, by, bw, bh, r=12, fill_color=colors.HexColor("#0D1528"), stroke=0)

    vals = [float(v) for v in fcf_path]
    vmin = min(vals + [0])
    vmax = max(vals + [0])
    span2 = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
    zero_y = by + 10 + (0 - vmin) / span2 * (bh - 20)
    c.setStrokeColor(BORDER); c.setLineWidth(1)
    c.line(bx + 10, zero_y, bx + bw - 10, zero_y)

    n = len(vals)
    gapb = 2
    barw = max(2, (bw - 20 - gapb * (n - 1)) / n)
    cx = bx + 10
    for v in vals:
        h = (abs(v) / span2) * (bh - 20)
        if v >= 0:
            c.setFillColor(ACCENT)
            c.rect(cx, zero_y, barw, h, fill=1, stroke=0)
        else:
            c.setFillColor(BAD)
            c.rect(cx, zero_y - h, barw, h, fill=1, stroke=0)
        cx += barw + gapb

    small(c, bx + 10, by + bh + 4, "FCF por ciclo (TV se reporta separado)", size=8.4)

    # table
    tx = left_x + 14
    ty = bottom_y + 74
    small(c, tx, ty + 42, "Ciclo", size=8.6)
    small(c, tx + 44, ty + 42, "FCF", size=8.6)

    rows = min(n_years, 8)
    for i in range(rows):
        text(c, tx, ty + 28 - i * 12, str(i + 1), size=9.2, bold=True, col=TEXT)
        small(c, tx + 44, ty + 28 - i * 12, money_pyg(vals[i]), size=9.2, col=TEXT)

    small(c, tx, bottom_y + 16, f"Valor Terminal (fin ciclo {n_years}): {money_pyg(float(tv))}", size=9.2, col=MUTED)

    # Risk (right bottom)
    rr(c, right_x, bottom_y, col_w, bottom_h, r=16, fill_color=CARD)
    text(c, right_x + 14, bottom_y + bottom_h - 22, "Riesgo (Monte Carlo) – lectura ejecutiva", size=10.8, bold=True)
    c.line(right_x + 14, bottom_y + bottom_h - 30, right_x + col_w - 14, bottom_y + bottom_h - 30)

    rx = right_x + 14
    ry = bottom_y + bottom_h - 50
    text(c, rx, ry, f"P50(VAN): {money_pyg(p50)}", size=10.6, bold=True, col=TEXT); ry -= 14
    small(c, rx, ry, f"P5(VAN): {money_pyg(p5)} · P95(VAN): {money_pyg(p95)}", size=9.2); ry -= 14
    small(c, rx, ry, f"P(VAN<0): {prob_neg:.1%} · Simulaciones: {sims:,}", size=9.2); ry -= 10
    if driver_focus:
        small(c, rx, ry, f"Variable más sensible: {driver_focus[:64] + ('…' if len(driver_focus) > 64 else '')}", size=9.0); ry -= 10

    # Mini hist
    hx2 = right_x + 14
    hy2 = bottom_y + 14
    hw2 = col_w - 28
    hh2 = 1.32 * inch
    rr(c, hx2, hy2, hw2, hh2, r=12, fill_color=colors.HexColor("#0D1528"), stroke=0)

    if mc_samples is None or mc_samples.size < 200:
        small(c, hx2 + 10, hy2 + hh2 / 2, "Histograma no disponible (Monte Carlo desactivado o pocas muestras)", size=9.0)
    else:
        arr = np.asarray(mc_samples)
        lo2, hi2 = float(np.min(arr)), float(np.max(arr))
        if lo2 == hi2:
            lo2 -= 1
            hi2 += 1
        bins = 18
        counts = np.zeros(bins, dtype=int)
        for v in arr:
            k = int((v - lo2) / (hi2 - lo2) * bins)
            k = max(0, min(bins - 1, k))
            counts[k] += 1
        m = int(np.max(counts)) if int(np.max(counts)) != 0 else 1
        barw2 = (hw2 - 20 - (bins - 1)) / bins
        cx = hx2 + 10
        for ct in counts:
            bh2 = (ct / m) * (hh2 - 20)
            c.setFillColor(ACCENT)
            c.rect(cx, hy2 + 8, barw2, bh2, fill=1, stroke=0)
            cx += barw2 + 1

        def x_of(v):
            return hx2 + 10 + (v - lo2) / (hi2 - lo2) * (hw2 - 20)

        for v, col in [(p5, BAD), (p50, WARN), (p95, GOOD)]:
            c.setStrokeColor(col); c.setLineWidth(1.2)
            xv = x_of(v)
            c.line(xv, hy2 + 8, xv, hy2 + hh2 - 8)

        small(c, hx2 + 10, hy2 + hh2 + 4, "Distribución VAN (P5 rojo · P50 ámbar · P95 verde)", size=8.4)

    # Footer
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8.2)
    c.drawString(margin, 0.45 * inch,
                 "Uso académico (MBA). Resultados dependen de supuestos y evidencia; no sustituyen due diligence.")
    c.drawRightString(W - margin, 0.45 * inch, "ValuationApp – One-Pager Dashboard (PYG)")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()

# ============================================================
# Sidebar (institucional + inputs)
# ============================================================
st.sidebar.header("🏛️ Encabezado institucional")
institution = st.sidebar.text_input("Institución", "Universidad San Ignacio de Loyola (USIL)")
program = st.sidebar.text_input("Programa", "Maestría en Administración de Negocios (MBA)")
course = st.sidebar.text_input("Curso / Módulo", "Proyectos de Inversión / Valuation")

st.sidebar.divider()
st.sidebar.header("🧩 Identificación")
project = st.sidebar.text_input("Proyecto", "Proyecto")
responsible = st.sidebar.text_input("Responsable", "Docente: Jorge Rojas")

st.sidebar.divider()
st.sidebar.header("🇵🇾 Contexto")
st.sidebar.markdown("**Moneda:** PYG (Gs.)")
st.sidebar.caption("Se asume base nominal. D/E se considera contable (práctica predominante en Paraguay).")

st.sidebar.divider()
st.sidebar.header("🖼️ Logo PDF (opcional)")
logo_file = st.sidebar.file_uploader("Subir logo (PNG/JPG)", type=["png", "jpg", "jpeg"])
logo_url = st.sidebar.text_input("o URL directa de imagen (.png/.jpg)", value="").strip()
logo_reader = _load_logo_reader(logo_file, logo_url if logo_url else None)

# --- Inversión ---
st.sidebar.divider()
st.sidebar.header("0) Inversión inicial")
capex0 = st.sidebar.number_input("CAPEX Año 0 (Gs.)", value=3500000000.0, step=50000000.0, min_value=1.0)

# --- CAPM/WACC ---
st.sidebar.divider()
st.sidebar.header("1) CAPM / WACC")
rf = st.sidebar.number_input("Tasa libre de riesgo Rf (%)", value=4.5, step=0.1) / 100
erp = st.sidebar.number_input("Prima de riesgo de mercado (ERP %) ", value=5.5, step=0.1) / 100
use_crp = st.sidebar.checkbox("Incluir Riesgo País (CRP) en Ke", value=True)
crp = st.sidebar.number_input("CRP (%)", value=2.0, step=0.1) / 100 if use_crp else 0.0
beta_u = st.sidebar.number_input("βU (desapalancada)", value=0.90, step=0.05)
tax_rate = st.sidebar.number_input("Impuesto T (%)", value=10.0, step=0.5) / 100

# --- Estructura capital contable ---
st.sidebar.divider()
st.sidebar.header("2) Estructura de capital (contable) para WACC")
debt = st.sidebar.number_input("Deuda D (contable, Gs.)", value=2800000000.0, step=50000000.0, min_value=0.0)
equity = st.sidebar.number_input("Capital propio E (contable, Gs.)", value=4200000000.0, step=50000000.0, min_value=1.0)
kd = st.sidebar.number_input("Costo de deuda Kd (%)", value=7.0, step=0.1) / 100

beta_l = beta_u * (1.0 + (1.0 - tax_rate) * (debt / equity))
ke = rf + beta_l * erp + (crp if use_crp else 0.0)
total_cap = debt + equity
wacc = (equity / total_cap) * ke + (debt / total_cap) * kd * (1.0 - tax_rate)

# --- Proyecciones DCF ---
st.sidebar.divider()
st.sidebar.header("3) Proyecciones (DCF)")
n_years = st.sidebar.slider("Años de proyección explícita (Ciclo 1…N)", 1, 10, 5)
g_exp = st.sidebar.number_input("g (crecimiento explícito %) ", value=5.0, step=0.25) / 100
g_inf = st.sidebar.number_input("g∞ (crecimiento perpetuo %) ", value=2.0, step=0.10) / 100
st.sidebar.caption("TV requiere WACC > g∞. En Monte Carlo: WACC > g∞ + 0.5%.")

# --- FCF Año 1 ---
st.sidebar.divider()
st.sidebar.header("4) Año 1: construcción del FCF")
fcf_mode = st.sidebar.radio(
    "Método para definir FCF Año 1",
    ["Ingresar FCF directamente", "Construir FCF desde cifras contables (Año 1)"],
    index=1
)

bridge_rows = None
if fcf_mode == "Ingresar FCF directamente":
    fcf_y1 = st.sidebar.number_input("FCF Año 1 (Gs.)", value=700000000.0, step=20000000.0, min_value=0.0)
else:
    st.sidebar.caption("Puente contable → caja: NOPAT + Depreciación − CAPEX mant. − ΔNWC (desagregado).")
    sales_y1 = st.sidebar.number_input("Ventas Año 1 (Gs.)", value=8000000000.0, step=100000000.0, min_value=0.0)
    var_costs_y1 = st.sidebar.number_input("Costos variables Año 1 (Gs.)", value=3200000000.0, step=100000000.0, min_value=0.0)
    fixed_costs_y1 = st.sidebar.number_input("Costos fijos Año 1 (Gs.)", value=2200000000.0, step=100000000.0, min_value=0.0)
    depreciation_y1 = st.sidebar.number_input("Depreciación Año 1 (Gs.)", value=300000000.0, step=10000000.0, min_value=0.0)
    maint_capex_y1 = st.sidebar.number_input("CAPEX (mantenimiento) Año 1 (Gs.)", value=250000000.0, step=10000000.0, min_value=0.0)

    st.sidebar.subheader("Δ Capital de Trabajo – Componentes")
    st.sidebar.caption("ΔNWC ≈ ΔCxC + ΔInventarios − ΔCxP. Positivo consume caja; negativo libera caja.")
    delta_ar = st.sidebar.number_input("Δ Cuentas por Cobrar (ΔCxC) (Gs.)", value=120000000.0, step=5000000.0)
    delta_inv = st.sidebar.number_input("Δ Inventarios (Gs.)", value=80000000.0, step=5000000.0)
    delta_ap = st.sidebar.number_input("Δ Cuentas por Pagar (ΔCxP) (Gs.)", value=60000000.0, step=5000000.0)
    delta_nwc_y1 = (delta_ar + delta_inv - delta_ap)

    other_cash_y1 = st.sidebar.number_input("Otros ajustes de caja Año 1 (opcional, Gs.)", value=0.0, step=5000000.0)

    ebitda_y1 = sales_y1 - var_costs_y1 - fixed_costs_y1
    ebit_y1 = ebitda_y1 - depreciation_y1
    nopat_y1 = ebit_y1 * (1.0 - tax_rate)

    fcf_y1 = nopat_y1 + depreciation_y1 - maint_capex_y1 - delta_nwc_y1 + other_cash_y1

    bridge_rows = [
        ("Ventas", sales_y1),
        ("- Costos variables", -var_costs_y1),
        ("- Costos fijos", -fixed_costs_y1),
        ("EBITDA", ebitda_y1),
        ("- Depreciación (no caja)", -depreciation_y1),
        ("EBIT", ebit_y1),
        (f"NOPAT = EBIT*(1-T), T={tax_rate:.0%}", nopat_y1),
        ("+ Depreciación (se revierte)", depreciation_y1),
        ("- CAPEX mantenimiento", -maint_capex_y1),
        ("- Δ Cuentas por Cobrar", -delta_ar),
        ("- Δ Inventarios", -delta_inv),
        ("+ Δ Cuentas por Pagar", +delta_ap),
        ("- Δ Capital de Trabajo (ΔNWC)", -delta_nwc_y1),
        ("+ Otros ajustes de caja", other_cash_y1),
        ("FCF Año 1 (resultado)", fcf_y1),
    ]

# --- Monte Carlo ---
st.sidebar.divider()
st.sidebar.header("5) Monte Carlo")
use_mc = st.sidebar.checkbox("Activar simulación Monte Carlo", value=True)
sims = st.sidebar.slider("Simulaciones", 1000, 50000, 10000, 1000)

st.sidebar.subheader("Rangos triangulares – g (explícito)")
g_min = st.sidebar.number_input("g mínimo (%)", value=1.0, step=0.25) / 100
g_mode = st.sidebar.number_input("g más probable (%)", value=5.0, step=0.25) / 100
g_max = st.sidebar.number_input("g máximo (%)", value=9.0, step=0.25) / 100

st.sidebar.subheader("Rangos triangulares – CAPEX (Año 0)")
capex_min = st.sidebar.number_input("CAPEX mínimo (Gs.)", value=3200000000.0, step=50000000.0, min_value=1.0)
capex_mode = st.sidebar.number_input("CAPEX más probable (Gs.)", value=3500000000.0, step=50000000.0, min_value=1.0)
capex_max = st.sidebar.number_input("CAPEX máximo (Gs.)", value=4200000000.0, step=50000000.0, min_value=1.0)

st.sidebar.subheader("Shock al nivel de FCF Año 1 (multiplicador)")
st.sidebar.caption("Representa shocks de demanda/margen. Ej.: 0.90 = -10%, 1.10 = +10%.")
fcf_mult_min = st.sidebar.number_input("Multiplicador mínimo", value=0.85, step=0.01, min_value=0.01)
fcf_mult_mode = st.sidebar.number_input("Multiplicador más probable", value=1.00, step=0.01, min_value=0.01)
fcf_mult_max = st.sidebar.number_input("Multiplicador máximo", value=1.15, step=0.01, min_value=0.01)

st.sidebar.subheader("Rangos triangulares – WACC")
auto_wacc = st.sidebar.checkbox("Auto WACC (±2% absoluto alrededor del WACC calculado)", value=True)
if not auto_wacc:
    w_min = st.sidebar.number_input("WACC mínimo (%)", value=9.0, step=0.25) / 100
    w_mode = st.sidebar.number_input("WACC más probable (%)", value=11.0, step=0.25) / 100
    w_max = st.sidebar.number_input("WACC máximo (%)", value=13.0, step=0.25) / 100
else:
    w_min = w_mode = w_max = None

# ============================================================
# Cálculo DCF determinístico
# ============================================================
years = np.arange(1, n_years + 1)

if wacc <= g_inf:
    st.error("Condición inválida: WACC debe ser mayor que g∞ para calcular el valor terminal (Gordon-Shapiro).")
    st.stop()

fcf_path = np.array([fcf_y1 * (1.0 + g_exp) ** (t - 1) for t in years], dtype=float)
tv = (fcf_path[-1] * (1.0 + g_inf)) / (wacc - g_inf)

pv_fcf = float(np.sum(fcf_path / ((1.0 + wacc) ** years)))
pv_tv = float(tv / ((1.0 + wacc) ** n_years))
pv_total = pv_fcf + pv_tv
base_npv = pv_total - capex0

cash_for_irr = np.concatenate(([-capex0], fcf_path[:-1], [fcf_path[-1] + tv]))
irr_raw = npf.irr(cash_for_irr)
base_irr = safe_irr(irr_raw)
nonconv = detect_non_conventional_flows(cash_for_irr)

flows_for_payback = np.concatenate((fcf_path[:-1], [fcf_path[-1] + tv]))
pb_year, pb_years = compute_discounted_payback(capex0, flows_for_payback, wacc)
if pb_year is None:
    payback_text = "No recupera en el horizonte (descontado)"
else:
    payback_text = f"≈ {pb_years:.2f} años (descontado)" if pb_years is not None else f"Ciclo {pb_year} (descontado)"

# ============================================================
# Monte Carlo + Comité
# ============================================================
mc_samples = None
p5 = p50 = p95 = 0.0
prob_neg = 0.0
verdict, rationale = "N/A", "Active Monte Carlo para dictamen."
driver_focus = None
actions = ["Active Monte Carlo para dictamen."]

if use_mc:
    if not tri_ok(g_min, g_mode, g_max):
        st.error("Rango triangular inválido para g: g_min ≤ g_mode ≤ g_max.")
        st.stop()
    if not tri_ok(capex_min, capex_mode, capex_max):
        st.error("Rango triangular inválido para CAPEX: min ≤ mode ≤ max.")
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
            st.error("Rango triangular inválido para WACC: min ≤ mode ≤ max.")
            st.stop()
        w_min_mc, w_mode_mc, w_max_mc = w_min, w_mode, w_max

    if w_min_mc <= g_inf + MIN_SPREAD:
        w_min_mc = g_inf + MIN_SPREAD + 0.0005
        w_mode_mc = max(w_mode_mc, g_inf + MIN_SPREAD + 0.0010)
        w_max_mc = max(w_max_mc, g_inf + MIN_SPREAD + 0.0015)

    npv_s, g_s, w_s, capex_s, mult_s, idx = run_monte_carlo(
        sims=int(sims),
        fcf_y1=float(fcf_y1),
        n_years=int(n_years),
        g_inf=float(g_inf),
        min_spread=MIN_SPREAD,
        g_min=float(g_min), g_mode=float(g_mode), g_max=float(g_max),
        w_min=float(w_min_mc), w_mode=float(w_mode_mc), w_max=float(w_max_mc),
        capex_min=float(capex_min), capex_mode=float(capex_mode), capex_max=float(capex_max),
        fcf_mult_min=float(fcf_mult_min), fcf_mult_mode=float(fcf_mult_mode), fcf_mult_max=float(fcf_mult_max),
        seed=MC_SEED,
    )

    valid_npvs = npv_s[~np.isnan(npv_s)]
    if valid_npvs.size < 200:
        st.error("Muy pocas simulaciones válidas. Revise rangos de WACC y g∞ (condición WACC > g∞ + spread).")
        st.stop()

    mc_samples = valid_npvs

    p5 = float(np.percentile(valid_npvs, 5))
    p50 = float(np.percentile(valid_npvs, 50))
    p95 = float(np.percentile(valid_npvs, 95))
    prob_neg = float(np.mean(valid_npvs < 0))

    mask = ~np.isnan(npv_s)
    corrs = {
        "g (crecimiento explícito)": safe_corr(npv_s[mask], g_s[mask]),
        "WACC": safe_corr(npv_s[mask], w_s[mask]),
        "CAPEX": safe_corr(npv_s[mask], capex_s[mask]),
        "Shock FCF Año 1 (multiplicador)": safe_corr(npv_s[mask], mult_s[mask]),
    }
    driver_focus = max(corrs.items(), key=lambda kv: (0 if np.isnan(kv[1]) else abs(kv[1])))[0]

    verdict, rationale = committee_verdict(prob_neg, p50, p5)
    actions = recommended_actions(prob_neg, p5, p50, driver_focus)

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Resumen",
    "📄 Detalle DCF",
    "🧾 Puente contable → FCF",
    "🎲 Monte Carlo + Comité",
    "📥 One-Pager PDF"
])

# ============================================================
# TAB 1 – Resumen
# ============================================================
with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("WACC", pct(wacc))
    c2.metric("VAN (caso base)", money_pyg(base_npv))
    c3.metric("TIR (caso base)", pct(base_irr) if base_irr is not None else "N/A")
    c4.metric("Payback (descont.)", payback_text)
    c5.metric("TV (fin ciclo N)", money_pyg(tv))

    if base_npv < 0:
        st.error("El VAN (caso base) es negativo: bajo supuestos centrales, el proyecto no crea valor.")
    else:
        st.success("El VAN (caso base) es positivo: bajo supuestos centrales, el proyecto crea valor.")

    if nonconv:
        st.warning("Se detectaron flujos no convencionales. La TIR puede no ser única; para comité se prioriza VAN + riesgo.")

    st.markdown("### Flujos explícitos (Ciclo 1…N)")
    df_flows = pd.DataFrame({"Ciclo": years, "FCF": fcf_path})
    st.plotly_chart(px.bar(df_flows, x="Ciclo", y="FCF", title="FCF explícito (sin TV)"), use_container_width=True)

    st.info(
        "**Gobernanza de cifras:** el número principal de decisión es el **VAN (caso base)**. "
        "Monte Carlo se presenta como bloque de riesgo (P5/P50/P95 y P(VAN<0))."
    )

# ============================================================
# TAB 2 – Detalle DCF
# ============================================================
with tab2:
    st.subheader("Detalle explícito del DCF (Ciclo 1…N + TV)")
    disc = 1.0 / ((1.0 + wacc) ** years)
    df_dcf = pd.DataFrame({
        "Ciclo (t)": years,
        "FCF proyectado": fcf_path,
        "Factor de descuento": disc,
        "FCF descontado (PV)": fcf_path * disc,
    })
    st.dataframe(df_dcf, use_container_width=True)

    st.markdown("### Valor terminal (crecimiento a perpetuidad g∞)")
    st.write(f"**g∞:** {pct(g_inf)}")
    st.write(f"**TV (fin del ciclo {n_years}):** {money_pyg(tv)}")
    st.write(f"**PV(TV):** {money_pyg(pv_tv)}")

    st.markdown("### Síntesis (transparencia total)")
    st.write(f"**PV(FCF 1…N):** {money_pyg(pv_fcf)}")
    st.write(f"**PV(TV):** {money_pyg(pv_tv)}")
    st.write(f"**CAPEX:** {money_pyg(capex0)}")
    st.success(f"**VAN (caso base): {money_pyg(base_npv)}**")

    st.download_button(
        "📥 Descargar detalle DCF (CSV)",
        data=df_dcf.to_csv(index=False).encode("utf-8"),
        file_name="detalle_dcf_pyg.csv",
        mime="text/csv"
    )

# ============================================================
# TAB 3 – Puente contable → FCF
# ============================================================
with tab3:
    st.subheader("Puente contable → FCF Año 1")
    if bridge_rows is None:
        st.info("FCF Año 1 ingresado directamente.")
        st.write(f"**FCF Año 1:** {money_pyg(fcf_y1)}")
    else:
        df_bridge = pd.DataFrame(bridge_rows, columns=["Concepto", "Monto"])
        st.dataframe(df_bridge, use_container_width=True)
        st.success(f"**FCF Año 1 estimado:** {money_pyg(fcf_y1)}")

        st.download_button(
            "📥 Descargar puente contable (CSV)",
            data=df_bridge.to_csv(index=False).encode("utf-8"),
            file_name="puente_contable_fcf_y1_pyg.csv",
            mime="text/csv"
        )

# ============================================================
# TAB 4 – Monte Carlo + Comité
# ============================================================
with tab4:
    st.subheader("Monte Carlo + dictamen de comité (criterios fijos)")
    st.caption("Criterios: P(VAN<0) ≤ 20%, P50>0, P5>0. Sin selectores para asegurar consistencia metodológica.")

    if not use_mc:
        st.info("Monte Carlo está desactivado en el sidebar.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P(VAN<0)", f"{prob_neg:.1%}")
        m2.metric("P5 (VAN)", money_pyg(p5))
        m3.metric("P50 (VAN)", money_pyg(p50))
        m4.metric("P95 (VAN)", money_pyg(p95))

        st.plotly_chart(px.histogram(x=mc_samples, nbins=40, title="Distribución del VAN simulado"), use_container_width=True)

        st.subheader(f"Dictamen: {verdict}")
        if verdict == "APROBADO":
            st.success(rationale)
        elif verdict == "OBSERVADO":
            st.warning(rationale)
        else:
            st.error(rationale)

        st.markdown("**Plan de acción**")
        for a in actions:
            st.write(f"- {a}")

        if driver_focus:
            st.caption(f"Sensibilidad orientativa (no causalidad): **{driver_focus}**")

# ============================================================
# TAB 5 – PDF One-Pager
# ============================================================
with tab5:
    st.subheader("One-Pager Ejecutivo (Dashboard premium – PYG)")
    st.caption("Este PDF reemplaza el reporte anterior de 2 páginas. VAN (caso base) es el número principal de decisión.")

    if not REPORTLAB_OK:
        st.warning("PDF deshabilitado: falta `reportlab` en requirements.txt.")
    else:
        try:
            pdf_bytes = generate_onepager_dashboard_pdf(
                institution=institution,
                program=program,
                course=course,
                project=project,
                responsible=responsible,
                n_years=n_years,
                wacc=wacc,
                g_exp=g_exp,
                g_inf=g_inf,
                capex0=capex0,
                fcf_path=fcf_path,
                tv=tv,
                van_base=base_npv,
                irr_base=base_irr,
                payback_text=payback_text,
                sims=int(sims) if use_mc else 0,
                prob_neg=prob_neg if use_mc else 0.0,
                p5=p5 if use_mc else 0.0,
                p50=p50 if use_mc else 0.0,
                p95=p95 if use_mc else 0.0,
                verdict=verdict if use_mc else "N/A",
                rationale=rationale if use_mc else "Active Monte Carlo para dictamen.",
                driver_focus=driver_focus if use_mc else None,
                mc_samples=mc_samples if use_mc else None,
                nonconv_warning=nonconv,
                logo_reader=logo_reader,
            )

            st.download_button(
                "📥 Descargar One-Pager (Dashboard Premium)",
                data=pdf_bytes,
                file_name=f"OnePager_Dashboard_{project.replace(' ', '_')}_PYG.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"No se pudo generar el PDF: {e}")

st.divider()
st.caption("Herramienta con fines académicos. La calidad del output depende de supuestos, evidencia y criterio profesional.")
