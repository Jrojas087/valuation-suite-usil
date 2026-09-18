# finanzas-personales/app.py
# Consultoría de Finanzas Personales — Test de riesgo + Diagnóstico rápido + Plan de acción + PDF
# Diplomado de Finanzas Personales — Herramienta independiente
# ------------------------------------------------------------
# Requisitos: ver requirements.txt en la raíz del repo (streamlit, reportlab)

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
.pill.na{ background: rgba(255,255,255,.06); opacity: .85; }
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

def rate_savings(rate: float, income: float) -> rep.Rating:
    if income <= 0:
        return rep.Rating("na", "No calculable", "⚪")
    if rate < 0.10:
        return rep.Rating("bad", "Bajo", "🔴")
    if rate < 0.20:
        return rep.Rating("warn", "Medio", "🟡")
    return rep.Rating("good", "Bueno", "🟢")

def rate_dti(dti: float, income: float) -> rep.Rating:
    if income <= 0:
        return rep.Rating("na", "No calculable", "⚪")
    if dti > 0.35:
        return rep.Rating("bad", "Alto", "🔴")
    if dti > 0.20:
        return rep.Rating("warn", "Moderado", "🟡")
    return rep.Rating("good", "Bajo", "🟢")

def rate_emergency(months: float, total_expenses: float) -> rep.Rating:
    # Igual que DTI/ahorro con ingreso=0: si no hay gastos totales con qué calcular
    # la cobertura, 0.0 no es "insuficiente" sino simplemente "no calculable".
    if total_expenses <= 0:
        return rep.Rating("na", "No calculable", "⚪")
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
st.sidebar.caption("Completalo al inicio de la sesión con tu cliente; aparece en el reporte final.")
consultant = st.sidebar.text_input(
    "Nombre del consultor/a (alumno/a)", "",
    placeholder="Ej: María Gómez",
    help="Tu nombre como alumno/a a cargo de esta consultoría. Aparece en el reporte.",
)
client = st.sidebar.text_input(
    "Nombre del cliente", "",
    placeholder="Ej: Juan Pérez",
    help="Nombre del cliente que estás asesorando. Aparece en el reporte.",
)
age = st.sidebar.number_input(
    "Edad del cliente", min_value=0, max_value=110, value=35, step=1,
    help="Usada como referencia para contextualizar el perfil de riesgo y el horizonte de planificación.",
)
occupation = st.sidebar.text_input("Ocupación", "", placeholder="Ej: Comerciante, docente, empleado/a")
dependents = st.sidebar.number_input(
    "N° de dependientes", min_value=0, max_value=20, value=0, step=1,
    help="Personas que dependen económicamente del cliente (hijos/as, familiares a cargo, etc.).",
)
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
    help="Elegí el motivo principal por el que el cliente busca esta consultoría; guía el plan de acción.",
)
st.sidebar.caption("🔒 No se guarda ningún dato: todo vive en esta sesión hasta que generes el reporte.")

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
        "<div class='small'>Leé cada afirmación en voz alta junto a tu cliente y marcá qué tan de acuerdo "
        "está con ella. No hay respuestas correctas o incorrectas: la idea es capturar su actitud real "
        "frente al riesgo, no lo que \"debería\" responder.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("ℹ️ Cómo usar la escala (mostrar antes de empezar)"):
        st.write(
            "La escala va de **Totalmente en desacuerdo** a **Totalmente de acuerdo**. Ninguna pregunta "
            "tiene una opción marcada por defecto: elegí una con el cliente para cada una. Se puede volver "
            "a cualquier pregunta antes de pasar a la pestaña de Diagnóstico."
        )
    answers = []
    for i, (question, invert) in enumerate(RISK_QUESTIONS, start=1):
        if i == 1:
            st.markdown("#### Bloque 1 · Actitud frente al riesgo y horizonte de inversión")
        if i == 6:
            st.markdown("")
            st.markdown("#### Bloque 2 · Reacciones emocionales y experiencia previa")
        st.markdown(f"**{i}. {question}**")
        choice = st.radio(
            f"Pregunta {i}", options=LIKERT_LABELS, index=None, key=f"risk_q_{i}",
            horizontal=True, label_visibility="collapsed",
        )
        if choice is None:
            answers.append(None)
        else:
            raw = LIKERT_LABELS.index(choice) + 1  # 1..5
            score = (6 - raw) if invert else raw
            answers.append(score)
        st.markdown("")
    st.markdown("</div>", unsafe_allow_html=True)

    answered_count = sum(1 for a in answers if a is not None)
    risk_complete = answered_count == len(RISK_QUESTIONS)
    risk_max = len(RISK_QUESTIONS) * 5

    st.markdown("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Resultado del test")
    if not risk_complete:
        risk_score = 0
        risk_category = "Pendiente"
        risk_desc = (
            f"Cuestionario incompleto: faltan {len(RISK_QUESTIONS) - answered_count} de "
            f"{len(RISK_QUESTIONS)} preguntas por responder. El perfil se calcula recién cuando "
            "las 10 están contestadas, para no asumir una actitud que el cliente no expresó."
        )
        alloc_fixed = alloc_variable = 0
        st.info(f"🕓 {answered_count}/{len(RISK_QUESTIONS)} preguntas respondidas — {risk_desc}")
    else:
        risk_score = int(sum(answers))
        risk_category, risk_desc, alloc_fixed, alloc_variable = classify_risk(risk_score)
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
    st.caption("Pedile al cliente montos aproximados; no hace falta precisión al peso, con un estimado alcanza.")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Ingresos y gastos**")
        income = st.number_input(
            "Ingreso mensual neto (todas las fuentes)", min_value=0.0, value=6_000_000.0, step=100_000.0,
            help="Sueldo neto, honorarios, rentas u otros ingresos regulares, sumados.",
        )
        fixed_expenses = st.number_input(
            "Gastos fijos mensuales (vivienda, servicios, seguros)", min_value=0.0, value=2_000_000.0, step=50_000.0,
            help="Alquiler o cuota de vivienda, luz, agua, internet, seguros y otros gastos que no varían mes a mes.",
        )
        variable_expenses = st.number_input(
            "Gastos variables mensuales (comida, transporte, ocio)", min_value=0.0, value=1_500_000.0, step=50_000.0,
            help="Supermercado, combustible o pasajes, salidas y otros gastos que cambian de mes a mes.",
        )
    with colB:
        st.markdown("**Deuda, ahorro y colchón**")
        debt_payment = st.number_input(
            "Cuota mensual total de deudas (tarjetas, préstamos)", min_value=0.0, value=500_000.0, step=50_000.0,
            help="Suma de todas las cuotas mensuales: tarjetas de crédito, préstamos personales, prendarios, etc.",
        )
        savings_monthly = st.number_input(
            "Ahorro/inversión mensual actual", min_value=0.0, value=500_000.0, step=50_000.0,
            help="Lo que el cliente realmente aparta o invierte cada mes, no lo que le \"sobra\".",
        )
        emergency_fund = st.number_input(
            "Fondo de emergencia actual (monto acumulado)", min_value=0.0, value=1_000_000.0, step=100_000.0,
            help="Ahorros líquidos y disponibles de inmediato ante un imprevisto (no incluye inversiones de largo plazo).",
        )
    net_worth_unknown = st.checkbox(
        "No cuento con este dato de patrimonio neto (dejar sin informar)",
        help="Marcá esta casilla si el cliente no sabe o no quiere compartir su patrimonio neto. "
        "Así se distingue de un patrimonio neto real de Gs. 0 (activos = pasivos).",
    )
    net_worth_input = st.number_input(
        "Patrimonio neto aproximado (activos − pasivos, opcional)", value=0.0, step=100_000.0,
        help="Opcional. Suma de todo lo que el cliente posee (ahorros, propiedades, vehículos) menos sus deudas totales.",
        disabled=net_worth_unknown,
    )
    # OJO: no usar "net_worth_input if net_worth_input != 0.0 else None" — eso hacía
    # indistinguible un patrimonio neto real de Gs. 0 de "el dato no se cargó". Ahora
    # "no informado" se decide explícitamente con el checkbox de arriba.
    net_worth = None if net_worth_unknown else net_worth_input
    st.markdown("</div>", unsafe_allow_html=True)

    total_expenses = fixed_expenses + variable_expenses + debt_payment
    cashflow = income - total_expenses
    savings_rate = safe_div(savings_monthly, income)
    dti = safe_div(debt_payment, income)
    emergency_months = safe_div(emergency_fund, (fixed_expenses + variable_expenses + debt_payment)) if (fixed_expenses + variable_expenses + debt_payment) else 0.0

    cashflow_rating = rate_cashflow(cashflow, income)
    savings_rating = rate_savings(savings_rate, income)
    dti_rating = rate_dti(dti, income)
    emergency_rating = rate_emergency(emergency_months, total_expenses)

    st.markdown("")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric(
        "Excedente antes de ahorro", fmt_pyg(cashflow),
        help="Ingreso menos gastos y cuotas de deuda, SIN restar el ahorro/inversión mensual. Si es negativo, el cliente gasta más de lo que gana.",
    )
    c1.markdown(pill_html(cashflow_rating), unsafe_allow_html=True)
    c2.metric(
        "Tasa de ahorro", fmt_pct(savings_rate) if savings_rating.key != "na" else "—",
        help="Ahorro mensual como % del ingreso. Referencia usada en esta app: menos de 10% es bajo, 10-20% medio, 20% o más se considera bueno.",
    )
    c2.markdown(pill_html(savings_rating), unsafe_allow_html=True)
    c3.metric(
        "Endeudamiento (DTI)", fmt_pct(dti) if dti_rating.key != "na" else "—",
        help="Cuotas de deuda como % del ingreso (Debt-to-Income). Cuanto más alto, menos margen financiero.",
    )
    c3.markdown(pill_html(dti_rating), unsafe_allow_html=True)
    c4.metric(
        "Fondo de emergencia", f"{emergency_months:.1f} meses",
        help="Cuántos meses de gastos totales cubre el fondo de emergencia actual ante una pérdida de ingresos.",
    )
    c4.markdown(pill_html(emergency_rating), unsafe_allow_html=True)

    available_after_savings = cashflow - savings_monthly
    st.caption(
        f"Disponible después de ahorro (excedente − ahorro declarado): **{fmt_pyg(available_after_savings)}**"
    )
    if available_after_savings < 0:
        st.warning(
            "⚠️ El ahorro/inversión mensual declarado (" + fmt_pyg(savings_monthly) + ") supera el excedente "
            "disponible (" + fmt_pyg(cashflow) + "). Verificá con el cliente si ese ahorro proviene de otros "
            "ingresos no declarados, de activos existentes o de deuda adicional."
        )

    diagnostic_ratings = [cashflow_rating, savings_rating, dti_rating, emergency_rating]
    health_points = sum(2 if r.key == "good" else (1 if r.key == "warn" else 0) for r in diagnostic_ratings)
    health_max = 8
    has_critical = any(r.key == "bad" for r in diagnostic_ratings)
    if health_points <= 3:
        health_label = "Situación crítica"
    elif health_points <= 5 or has_critical:
        # No mostramos "Sólida" si hay al menos un indicador en rojo (ej. fondo de
        # emergencia en 0), aunque el puntaje agregado sea alto: un solo indicador
        # crítico no debería quedar tapado por el promedio.
        health_label = "En desarrollo"
    else:
        health_label = "Sólida"

    st.markdown("")
    st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
    st.markdown(f"### Salud financiera general: **{health_label}** ({health_points}/{health_max})")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    action_plan = []
    if income <= 0:
        action_plan.append(
            "⚪ Antes que nada: completá el ingreso mensual del cliente. Sin ese dato no se puede calcular "
            "la tasa de ahorro ni el endeudamiento (quedan marcados como \"No calculable\")."
        )
    if cashflow_rating.key == "bad":
        action_plan.append(
            "🔴 Prioridad 1: cerrar el déficit mensual. Los gastos superan los ingresos; antes de ahorrar o "
            "invertir, ajusta el presupuesto (reduce gastos variables o incrementa ingresos)."
        )
    # Fondo de emergencia: se usa el rating (no el número crudo de meses) para no
    # tratar un "No calculable" (gastos totales = 0) como si fuera insuficiente.
    if emergency_rating.key == "bad":
        action_plan.append(
            f"🔴 Prioridad {len(action_plan)+1}: construir un fondo de emergencia de al menos 3 a 6 meses de "
            f"gastos (actualmente cubre {emergency_months:.1f} meses)."
        )
    elif emergency_rating.key == "warn":
        action_plan.append(
            f"🟡 Prioridad {len(action_plan)+1}: seguir reforzando el fondo de emergencia hasta cubrir 6 meses "
            f"de gastos (actualmente cubre {emergency_months:.1f} meses, cobertura parcial)."
        )
    if dti_rating.key == "bad":
        action_plan.append(
            f"🔴 Prioridad {len(action_plan)+1}: reducir el nivel de endeudamiento (actualmente "
            f"{dti*100:.0f}% del ingreso se destina a cuotas de deuda). Prioriza cancelar las deudas con mayor "
            f"tasa de interés."
        )
    elif dti_rating.key == "warn":
        action_plan.append(
            f"🟡 Prioridad {len(action_plan)+1}: el endeudamiento es moderado ({dti*100:.0f}% del ingreso). "
            f"Evita tomar nuevas deudas y buscá bajarlo hacia el 20% o menos."
        )
    if savings_rating.key == "bad" and cashflow >= 0:
        action_plan.append(
            f"🔴 Prioridad {len(action_plan)+1}: aumentar la tasa de ahorro mensual (actualmente "
            f"{savings_rate*100:.0f}%). Automatiza un porcentaje del ingreso apenas se recibe."
        )
    elif savings_rating.key == "warn" and cashflow >= 0:
        action_plan.append(
            f"🟡 Prioridad {len(action_plan)+1}: la tasa de ahorro es media ({savings_rate*100:.0f}%). Buscá "
            f"subirla gradualmente hacia 20% o más, por ejemplo automatizando un aumento cada vez que suba el "
            f"ingreso."
        )

    # El mensaje genérico de "todo en orden" solo debe aparecer cuando ningún
    # indicador quedó en rojo ni en amarillo (verde o "no calculable" están bien).
    all_indicators_ok = (
        cashflow_rating.key in ("good", "na")
        and emergency_rating.key in ("good", "na")
        and dti_rating.key in ("good", "na")
        and savings_rating.key in ("good", "na")
    )
    if not action_plan and all_indicators_ok:
        action_plan.append("🟢 Buen punto de partida: los indicadores básicos están en orden. El siguiente paso es definir metas concretas de mediano y largo plazo.")

    # Conecta el objetivo elegido en el sidebar con una recomendación concreta.
    if objective == "Reducir deudas":
        if dti_rating.key == "good":
            action_plan.append(
                "🎯 Objetivo del cliente — Reducir deudas: el endeudamiento (DTI) ya está en zona saludable, "
                "así que en vez de solo \"pagar más\" conviene revisar las tasas de interés de las deudas "
                "vigentes (refinanciar o consolidar las más caras) para liberar excedente mensual."
            )
        else:
            action_plan.append(
                "🎯 Objetivo del cliente — Reducir deudas: seguí el plan de pago de deudas indicado arriba "
                "antes de sumar compromisos financieros nuevos."
            )
    elif objective == "Empezar a invertir":
        if emergency_rating.key in ("bad", "na"):
            action_plan.append(
                "🎯 Objetivo del cliente — Empezar a invertir: antes de destinar dinero a inversiones conviene "
                "resolver el fondo de emergencia (hoy insuficiente o no calculable); invertir sin ese colchón "
                "expone al cliente a tener que vender en mal momento ante un imprevisto."
            )
        else:
            action_plan.append(
                "🎯 Objetivo del cliente — Empezar a invertir: con el fondo de emergencia en un nivel "
                "adecuado, el siguiente paso es definir la mezcla de inversión según el perfil de riesgo del "
                "cliente."
            )

    if risk_complete:
        action_plan.append(
            f"📈 Según el perfil de riesgo ({risk_category}), evalúa junto al cliente una mezcla ilustrativa acorde "
            f"antes de tomar decisiones de inversión concretas."
        )
    else:
        action_plan.append(
            "📈 Completá las 10 preguntas del test de perfil de riesgo (pestaña 1) para poder sugerir una "
            "mezcla de inversión ilustrativa acorde al cliente."
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
        "Revisá que los datos de las pestañas anteriores estén completos. Cuando estés listo/a, "
        "presioná \"Generar reporte\" para armar el documento con los valores actuales; recién ahí "
        "se habilitan los botones de descarga (no se generan solos en cada cambio de pestaña o "
        "casilla). No se guarda ninguna copia en el servidor ni en disco: el reporte generado vive "
        "solo en esta sesión del navegador, hasta que la cerrés o recargués la página."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    missing_fields = []
    if not consultant.strip():
        missing_fields.append("nombre del consultor/a")
    if not client.strip():
        missing_fields.append("nombre del cliente")
    if not risk_complete:
        missing_fields.append(f"cuestionario de perfil de riesgo ({answered_count}/{len(RISK_QUESTIONS)} respondidas)")
    if income <= 0:
        missing_fields.append("ingreso mensual del cliente")
    if missing_fields:
        st.warning(
            "⚠️ Falta completar: " + ", ".join(missing_fields) + ". "
            "Podés descargar igual, pero el reporte quedará incompleto en esas secciones."
        )

    st.markdown("")
    confirm_real_data = st.checkbox(
        "Confirmo que los valores ingresados en las pestañas anteriores corresponden a este cliente real "
        "(no son los montos de ejemplo precargados por la app).",
    )
    if not confirm_real_data:
        st.warning("☝️ Marcá la casilla de arriba para habilitar la generación del reporte.")

    if confirm_real_data:
        # Huella de los datos que entran al reporte: sirve para avisar si el alumno
        # generó el reporte y después siguió tocando inputs, sin recalcular nada en
        # cada rerun (eso es justamente lo que este botón evita).
        report_inputs_snapshot = (
            consultant, client, age, occupation, dependents, objective,
            risk_complete, risk_score, risk_max, risk_category, risk_desc,
            alloc_fixed, alloc_variable,
            income, fixed_expenses, variable_expenses, debt_payment,
            savings_monthly, emergency_fund, net_worth,
            tuple(action_plan),
        )

        if st.button("🔄 Generar reporte con los datos actuales"):
            report = rep.ClientFinReport(
                consultant=consultant or "—",
                client=client or "—",
                report_date=date.today().isoformat(),
                # OJO: no usar "if age" — edad=0 es un valor válido (aunque atípico) y una
                # comprobación de verdad lo convertiría incorrectamente en None ("—").
                age=int(age) if age is not None else None,
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
            # Recién acá se calculan el TXT y el PDF — no en cada rerun del script.
            st.session_state["fp_report_txt"] = rep.build_text_report(report)
            st.session_state["fp_report_pdf"] = (
                rep.generate_pdf(report) if rep.REPORTLAB_OK else None
            )
            st.session_state["fp_report_snapshot"] = report_inputs_snapshot

        report_ready = "fp_report_txt" in st.session_state
        if report_ready and st.session_state.get("fp_report_snapshot") != report_inputs_snapshot:
            st.info(
                "ℹ️ Los datos cambiaron desde la última vez que generaste el reporte. Volvé a "
                "presionar \"Generar reporte\" para que la descarga refleje los valores actuales."
            )

        if report_ready:
            st.download_button(
                "⬇️ Descargar reporte (TXT)",
                data=st.session_state["fp_report_txt"].encode("utf-8"),
                file_name="diagnostico_finanzas_personales.txt",
                mime="text/plain",
            )
            if st.session_state.get("fp_report_pdf") is not None:
                st.download_button(
                    "⬇️ Descargar reporte (PDF)",
                    data=st.session_state["fp_report_pdf"],
                    file_name="diagnostico_finanzas_personales.pdf",
                    mime="application/pdf",
                )
            elif not rep.REPORTLAB_OK:
                st.info("Para exportar PDF, agrega `reportlab` a requirements.txt.")
        else:
            st.info(
                "Presioná \"🔄 Generar reporte con los datos actuales\" para habilitar la descarga."
            )

st.sidebar.divider()
st.sidebar.subheader("📶 Progreso de la consultoría")
st.sidebar.write(f"1️⃣ Perfil de riesgo: **{risk_category}** ({risk_score}/{risk_max})")
st.sidebar.write(f"2️⃣ Diagnóstico financiero: **{health_label}** ({health_points}/{health_max})")
st.sidebar.write(f"3️⃣ Plan de acción: **{len(action_plan)}** recomendación(es)")
if "fp_report_txt" in st.session_state:
    st.sidebar.write("4️⃣ Reporte: ✅ generado — listo para descargar")
elif confirm_real_data:
    st.sidebar.write("4️⃣ Reporte: ⚠️ presioná \"Generar reporte\" en la pestaña Reporte")
else:
    st.sidebar.write("4️⃣ Reporte: ⚠️ confirmá los datos en la pestaña Reporte para generarlo")

st.markdown(
    "<div class='small' style='text-align:center; margin-top:8px;'>Uso educativo — Diplomado de Finanzas "
    "Personales. No constituye asesoría de inversión regulada.</div>",
    unsafe_allow_html=True,
)
