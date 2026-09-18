# pages/1_💰_Finanzas_Personales.py
# Consultoría de Finanzas Personales — Test de riesgo + Diagnóstico rápido + Plan de acción + PDF
# Diplomado de Finanzas Personales
# ------------------------------------------------------------
# Requisitos: ver requirements.txt (streamlit, reportlab — ya usados por la app de valuación)

from datetime import date

import streamlit as st

import report_finanzas_personales as rep

# ============================================================
# Config UI
# ============================================================
st.set_page_config(
    page_title="Finanzas Personales — Consultoría (PYG)",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
:root{
  --bg0:#050914; --bg1:#071026; --card:#0b1733; --card2:#0d1b3d;
  --line:rgba(255,255,255,.08); --text:#eaf1ff; --muted:rgba(234,241,255,.72);
  --accent:#66a9ff; --good:#27d17c; --warn:#ffcc66; --bad:#ff5d5d;
}
html, body, [class*="stApp"] { background: radial-gradient(1200px 700px at 20% 0%, #0a1736 0%, var(--bg0) 55%, #04070f 100%) !important; color: var(--text) !important; }
h1,h2,h3,h4,h5,h6, p, div, span, label { color: var(--text); }
[data-testid="stSidebar"]{ background: linear-gradient(180deg, var(--bg1), #050914) !important; border-right: 1px solid var(--line); }
[data-testid="stMetric"]{ background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02)); border: 1px solid var(--line); padding: 14px 14px; border-radius: 14px; }
.block-container { padding-top: 1.2rem; }
hr { border-color: var(--line); }
.card {
  background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
}
.card h3 { margin: 0 0 8px 0; font-size: 1.05rem; }
.small { color: var(--muted); font-size: .88rem; }
.pill {
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(102,169,255,.10);
  font-weight: 700; letter-spacing: .02em;
}
.pill.good{ background: rgba(39,209,124,.12); }
.pill.warn{ background: rgba(255,204,102,.12); }
.pill.bad{ background: rgba(255,93,93,.12); }
.kpiSub { color: var(--muted); font-size: .82rem; margin-top: -2px; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

st.title("💰 Consultoría de Finanzas Personales")
st.caption("Test de perfil de riesgo + diagnóstico financiero rápido + plan de acción. Herramienta educativa para alumnos del Diplomado. 🇵🇾")


# ============================================================
# Utilidades
# ============================================================
def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"

def fmt_pyg(x: float) -> str:
    return "Gs. {:,.0f}".format(float(x)).replace(",", ".")

def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0

def rate_cashflow(flow: float, income: float) -> rep.Rating:
    if flow < 0:
        return rep.Rating("bad", "Déficit", "🔴")
    ratio = safe_div(flow, income)
    if ratio < 0.10:
        return rep.Rating("warn", "Ajustado", "🟡")
    return rep.Rating("good", "Saludable", "🟢")

def rate_savings(rate: float) -> rep.Rating:
    if rate < 0.10:
        return rep.Rating("bad", "Bajo", "🔴")
    if rate < 0.20:
        return rep.Rating("warn", "Medio", "🟡")
    return rep.Rating("good", "Bueno", "🟢")

def rate_dti(dti: float) -> rep.Rating:
    if dti > 0.35:
        return rep.Rating("bad", "Alto", "🔴")
    if dti > 0.20:
        return rep.Rating("warn", "Moderado", "🟡")
    return rep.Rating("good", "Bajo", "🟢")

def rate_emergency(months: float) -> rep.Rating:
    if months < 3:
        return rep.Rating("bad", "Insuficiente", "🔴")
    if months < 6:
        return rep.Rating("warn", "Parcial", "🟡")
    return rep.Rating("good", "Adecuado", "🟢")

def pill_html(rating: rep.Rating) -> str:
    return f'<span class="pill {rating.key}">{rating.emoji} {rating.label}</span>'


# ============================================================
# Sidebar — Identificación
# ============================================================
st.sidebar.header("🧩 Identificación")
consultant = st.sidebar.text_input("Nombre del consultor/a (alumno/a)", "")
client = st.sidebar.text_input("Nombre del cliente", "")
age = st.sidebar.number_input("Edad del cliente", min_value=0, max_value=110, value=35, step=1)
occupation = st.sidebar.text_input("Ocupación", "")
dependents = st.sidebar.number_input("N° de dependientes", min_value=0, max_value=20, value=0, step=1)
objective = st.sidebar.selectbox(
    "Objetivo principal de la consulta",
    [
        "Ordenar sus finanzas",
        "Reducir deudas",
        "Construir un fondo de emergencia",
        "Planificar el retiro",
        "Comprar una vivienda",
        "Empezar a invertir",
        "Otro",
    ],
)
st.sidebar.caption("No se guarda ningún dato: todo vive en esta sesión hasta que generes el reporte.")

# ============================================================
# Preguntas del test de perfil de riesgo
# ============================================================
LIKERT_LABELS = ["Totalmente en desacuerdo", "En desacuerdo", "Neutral", "De acuerdo", "Totalmente de acuerdo"]

RISK_QUESTIONS = [
    ("Estoy dispuesto/a a invertir en productos que pueden subir o bajar de valor con tal de obtener una mayor rentabilidad a largo plazo.", False),
    ("Mi horizonte de inversión (tiempo antes de necesitar el dinero) es de más de 5 años.", False),
    ("Tengo experiencia previa invirtiendo en acciones, fondos mutuos u otros instrumentos de renta variable.", False),
    ("Mis ingresos mensuales son estables y predecibles.", False),
    ("Si mi inversión perdiera un 20% de su valor en un mes, me mantendría tranquilo/a y no vendería de inmediato.", False),
    ("Prefiero la seguridad de mi dinero antes que la posibilidad de obtener mayores ganancias.", True),
    ("Me sentiría muy angustiado/a si viera que mis ahorros pierden valor, aunque sea temporalmente.", True),
    ("Cuento con un fondo de emergencia que cubre mis gastos por varios meses.", False),
    ("Estaría dispuesto/a a destinar una parte importante de mis ahorros a inversiones de alto potencial de crecimiento, aunque impliquen mayor riesgo.", False),
    ("En decisiones financieras pasadas, he preferido opciones conocidas y de bajo riesgo, aunque el retorno fuera menor.", True),
]

RISK_PROFILES = [
    (18, "Conservador", "Prioriza la protección del capital sobre la rentabilidad. Prefiere instrumentos de bajo riesgo (ahorro, depósitos a plazo, bonos de alta calidad) y evita la volatilidad.", 80, 20),
    (26, "Moderado", "Busca cierto crecimiento del capital pero con baja tolerancia a pérdidas grandes. Prefiere una mezcla conservadora con algo de exposición a renta variable.", 65, 35),
    (34, "Balanceado", "Busca equilibrio entre crecimiento y protección. Tolera fluctuaciones moderadas en el corto plazo a cambio de mejores retornos en el largo plazo.", 50, 50),
    (42, "Crecimiento", "Prioriza el crecimiento del capital en el largo plazo y tolera fluctuaciones significativas en el corto/mediano plazo.", 35, 65),
    (50, "Agresivo", "Busca maximizar la rentabilidad de largo plazo, tolera alta volatilidad y posibles pérdidas temporales importantes con tal de obtener mayores retornos.", 20, 80),
]

def classify_risk(score: int):
    for limit, name, desc, fixed, variable in RISK_PROFILES:
        if score <= limit:
            return name, desc, fixed, variable
    return RISK_PROFILES[-1][1:]


# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Perfil de riesgo", "🩺 Diagnóstico financiero", "📋 Plan de acción", "📄 Reporte"]
)

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Cuestionario de tolerancia al riesgo")
    st.markdown(
        "<div class='small'>Lee cada afirmación junto a tu cliente y marca qué tan de acuerdo está. "
        "No hay respuestas correctas o incorrectas.</div>",
        unsafe_allow_html=True,
    )
    answers = []
    for i, (question, invert) in enumerate(RISK_QUESTIONS, start=1):
        st.markdown(f"**{i}. {question}**")
        choice = st.select_slider(
            f"risk_q_{i}", options=LIKERT_LABELS, value="Neutral", key=f"risk_q_{i}", label_visibility="collapsed"
        )
        raw = LIKERT_LABELS.index(choice) + 1  # 1..5
        score = (6 - raw) if invert else raw
        answers.append(score)
        st.markdown("")
    st.markdown("</div>", unsafe_allow_html=True)

    risk_score = int(sum(answers))
    risk_max = len(RISK_QUESTIONS) * 5
    risk_category, risk_desc, alloc_fixed, alloc_variable = classify_risk(risk_score)

    st.markdown("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Resultado del test")
    c1, c2 = st.columns([0.3, 0.7])
    with c1:
        st.metric("Puntaje", f"{risk_score} / {risk_max}")
        st.markdown(f"**Perfil: {risk_category}**")
    with c2:
        st.write(risk_desc)
        st.caption(
            f"Mezcla ilustrativa (no es recomendación de inversión): "
            f"{alloc_fixed}% bajo riesgo/ahorro — {alloc_variable}% mayor riesgo/crecimiento."
        )
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Datos financieros mensuales del cliente (Gs.)")
    colA, colB = st.columns(2)
    with colA:
        income = st.number_input("Ingreso mensual neto (todas las fuentes)", min_value=0.0, value=6_000_000.0, step=100_000.0)
        fixed_expenses = st.number_input("Gastos fijos mensuales (vivienda, servicios, seguros)", min_value=0.0, value=2_000_000.0, step=50_000.0)
        variable_expenses = st.number_input("Gastos variables mensuales (comida, transporte, ocio)", min_value=0.0, value=1_500_000.0, step=50_000.0)
    with colB:
        debt_payment = st.number_input("Cuota mensual total de deudas (tarjetas, préstamos)", min_value=0.0, value=500_000.0, step=50_000.0)
        savings_monthly = st.number_input("Ahorro/inversión mensual actual", min_value=0.0, value=500_000.0, step=50_000.0)
        emergency_fund = st.number_input("Fondo de emergencia actual (monto acumulado)", min_value=0.0, value=1_000_000.0, step=100_000.0)
    net_worth_input = st.number_input("Patrimonio neto aproximado (activos − pasivos, opcional)", value=0.0, step=100_000.0)
    net_worth = net_worth_input if net_worth_input != 0.0 else None
    st.markdown("</div>", unsafe_allow_html=True)

    total_expenses = fixed_expenses + variable_expenses + debt_payment
    cashflow = income - total_expenses
    savings_rate = safe_div(savings_monthly, income)
    dti = safe_div(debt_payment, income)
    emergency_months = safe_div(emergency_fund, (fixed_expenses + variable_expenses + debt_payment)) if (fixed_expenses + variable_expenses + debt_payment) else 0.0

    cashflow_rating = rate_cashflow(cashflow, income)
    savings_rating = rate_savings(savings_rate)
    dti_rating = rate_dti(dti)
    emergency_rating = rate_emergency(emergency_months)

    st.markdown("")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("Flujo de caja mensual", fmt_pyg(cashflow))
    c1.markdown(pill_html(cashflow_rating), unsafe_allow_html=True)
    c2.metric("Tasa de ahorro", fmt_pct(savings_rate))
    c2.markdown(pill_html(savings_rating), unsafe_allow_html=True)
    c3.metric("Endeudamiento (DTI)", fmt_pct(dti))
    c3.markdown(pill_html(dti_rating), unsafe_allow_html=True)
    c4.metric("Fondo de emergencia", f"{emergency_months:.1f} meses")
    c4.markdown(pill_html(emergency_rating), unsafe_allow_html=True)

    health_points = sum(2 if r.key == "good" else (1 if r.key == "warn" else 0) for r in [cashflow_rating, savings_rating, dti_rating, emergency_rating])
    health_max = 8
    if health_points <= 3:
        health_label = "Situación crítica"
    elif health_points <= 5:
        health_label = "En desarrollo"
    else:
        health_label = "Sólida"

    st.markdown("")
    st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
    st.markdown(f"### Salud financiera general: **{health_label}** ({health_points}/{health_max})")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    action_plan = []
    if cashflow < 0:
        action_plan.append(
            "🔴 Prioridad 1: cerrar el déficit mensual. Los gastos superan los ingresos; antes de ahorrar o "
            "invertir, ajusta el presupuesto (reduce gastos variables o incrementa ingresos)."
        )
    if emergency_months < 3:
        action_plan.append(
            f"🔴 Prioridad {len(action_plan)+1}: construir un fondo de emergencia de al menos 3 a 6 meses de "
            f"gastos (actualmente cubre {emergency_months:.1f} meses)."
        )
    if dti > 0.35:
        action_plan.append(
            f"🟡 Prioridad {len(action_plan)+1}: reducir el nivel de endeudamiento (actualmente "
            f"{dti*100:.0f}% del ingreso se destina a cuotas de deuda). Prioriza cancelar las deudas con mayor "
            f"tasa de interés."
        )
    if savings_rate < 0.10 and cashflow >= 0:
        action_plan.append(
            f"🟡 Prioridad {len(action_plan)+1}: aumentar la tasa de ahorro mensual (actualmente "
            f"{savings_rate*100:.0f}%). Automatiza un porcentaje del ingreso apenas se recibe."
        )
    if not action_plan:
        action_plan.append("🟢 Buen punto de partida: los indicadores básicos están en orden. El siguiente paso es definir metas concretas de mediano y largo plazo.")
    action_plan.append(
        f"📈 Según el perfil de riesgo ({risk_category}), evalúa junto al cliente una mezcla ilustrativa acorde "
        f"antes de tomar decisiones de inversión concretas."
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Plan de acción sugerido")
    for item in action_plan:
        st.write(item)
    st.markdown("</div>", unsafe_allow_html=True)

    st.info(rep.DISCLAIMER)

with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Generar reporte")
    st.write(
        "Revisa que los datos de las pestañas anteriores estén completos y luego descarga el reporte "
        "para entregar a tu cliente. No se guarda ninguna copia: solo se genera al hacer clic."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    report = rep.ClientFinReport(
        consultant=consultant or "—",
        client=client or "—",
        report_date=date.today().isoformat(),
        age=int(age) if age else None,
        occupation=occupation or "—",
        dependents=int(dependents),
        objective=objective,
        risk_score=risk_score,
        risk_max=risk_max,
        risk_category=risk_category,
        risk_description=risk_desc,
        risk_alloc_fixed=alloc_fixed,
        risk_alloc_variable=alloc_variable,
        income=float(income),
        fixed_expenses=float(fixed_expenses),
        variable_expenses=float(variable_expenses),
        debt_payment=float(debt_payment),
        savings_monthly=float(savings_monthly),
        emergency_fund=float(emergency_fund),
        net_worth=float(net_worth) if net_worth is not None else None,
        total_expenses=float(total_expenses),
        cashflow=float(cashflow),
        savings_rate=float(savings_rate),
        dti=float(dti),
        emergency_months=float(emergency_months),
        cashflow_rating=cashflow_rating,
        savings_rating=savings_rating,
        dti_rating=dti_rating,
        emergency_rating=emergency_rating,
        health_score=int(health_points),
        health_max=int(health_max),
        health_label=health_label,
        action_plan=action_plan,
    )

    txt_report = rep.build_text_report(report)
    st.download_button(
        "⬇️ Descargar reporte (TXT)",
        data=txt_report.encode("utf-8"),
        file_name="diagnostico_finanzas_personales.txt",
        mime="text/plain",
    )

    if rep.REPORTLAB_OK:
        pdf_bytes = rep.generate_pdf(report)
        st.download_button(
            "⬇️ Descargar reporte (PDF)",
            data=pdf_bytes,
            file_name="diagnostico_finanzas_personales.pdf",
            mime="application/pdf",
        )
    else:
        st.info("Para exportar PDF, agrega `reportlab` a requirements.txt.")

st.markdown(
    "<div class='small' style='text-align:center; margin-top:8px;'>Uso educativo — Diplomado de Finanzas "
    "Personales. No constituye asesoría de inversión regulada.</div>",
    unsafe_allow_html=True,
)
