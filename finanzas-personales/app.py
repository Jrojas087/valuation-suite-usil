# finanzas-personales/app.py
# Consultoría de Finanzas Personales — Test de riesgo + Diagnóstico rápido + Plan de acción + PDF
# Diplomado de Finanzas Personales — Herramienta independiente
# ------------------------------------------------------------
# Requisitos: ver requirements.txt en la raíz del repo (streamlit, reportlab)

import re
from datetime import date

import pandas as pd
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
    # OJO: "na" solo cuando el formulario está realmente vacío (sin ingreso NI
    # gastos, flow=0). Si income=0 pero flow<0 (hay gastos sin ingreso alguno),
    # es un déficit real y debe seguir en rojo — marcarlo "na" lo taparía.
    if income <= 0 and flow == 0:
        return rep.Rating("na", "No calculable", "⚪")
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

def strip_emoji_prefix(s: str) -> str:
    # Quita el emoji/símbolo inicial de los ítems del plan de acción (ej. "🔴 Prioridad
    # 1: ...") para usarlos como texto de partida más limpio en la tabla de compromisos.
    return re.sub(r"^[^\wÁÉÍÓÚÑáéíóúñ]+", "", s).strip()

def safe_cell_num(v, default=0.0) -> float:
    # En las tablas editables (data_editor), una celda numérica vacía llega como
    # NaN — y "NaN or 0" da NaN (no 0), porque NaN es "truthy" en Python. Hay que
    # chequear explícitamente con pd.notna(), no con el truthiness de la celda.
    return float(v) if pd.notna(v) else default

def safe_cell_text(v, default="") -> str:
    return str(v).strip() if pd.notna(v) else default


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
# Cartera modelo sugerida (por categoría de instrumento, no títulos
# específicos) — 3 niveles, mapeados desde las 5 categorías del test y
# ajustados por edad como proxy simple de CAPACIDAD de riesgo (no solo
# tolerancia/actitud). Es un modelo educativo e ilustrativo, no una
# recomendación de inversión personalizada.
# ============================================================
PORTFOLIO_MODELS = {
    "Conservadora": [
        ("Liquidez / Ahorro", 15),
        ("CDA (Certificados de Depósito de Ahorro)", 40),
        ("Bonos gubernamentales", 30),
        ("Bonos corporativos", 10),
        ("Renta variable (acciones/fondos)", 5),
    ],
    "Moderada": [
        ("Liquidez / Ahorro", 10),
        ("CDA (Certificados de Depósito de Ahorro)", 25),
        ("Bonos gubernamentales", 25),
        ("Bonos corporativos", 20),
        ("Renta variable (acciones/fondos)", 20),
    ],
    "Arriesgada": [
        ("Liquidez / Ahorro", 5),
        ("CDA (Certificados de Depósito de Ahorro)", 10),
        ("Bonos gubernamentales", 15),
        ("Bonos corporativos", 20),
        ("Renta variable (acciones/fondos)", 50),
    ],
}
PORTFOLIO_TIERS_ORDER = ["Conservadora", "Moderada", "Arriesgada"]

# Mapeo de las 5 categorías del test (tolerancia) a los 3 niveles de cartera.
PORTFOLIO_TIER_BY_RISK_CATEGORY = {
    "Conservador": "Conservadora",
    "Moderado": "Conservadora",
    "Balanceado": "Moderada",
    "Crecimiento": "Arriesgada",
    "Agresivo": "Arriesgada",
}

# Edad a partir de la cual se aplica el ajuste hacia un nivel más conservador.
# No se sube de nivel para clientes jóvenes: el puntaje del test ya define el
# techo de riesgo tolerado; la edad solo puede bajarlo (nunca subirlo), porque
# un horizonte largo no vuelve "más tolerante" a alguien que no lo es.
PORTFOLIO_AGE_CONSERVATIVE_THRESHOLD = 60

def suggest_portfolio(risk_category: str, age):
    """Devuelve (nivel, [(categoría, %), ...], nota_de_edad_o_None)."""
    base_tier = PORTFOLIO_TIER_BY_RISK_CATEGORY.get(risk_category)
    if base_tier is None:
        return None, [], None
    tier = base_tier
    age_note = None
    idx = PORTFOLIO_TIERS_ORDER.index(base_tier)
    if age is not None and age >= PORTFOLIO_AGE_CONSERVATIVE_THRESHOLD and idx > 0:
        tier = PORTFOLIO_TIERS_ORDER[idx - 1]
        age_note = (
            f"Ajustado un nivel más conservador que el resultado del test (que por sí solo sugería "
            f"'{base_tier}') por la edad del cliente ({int(age)} años): a mayor edad, menor horizonte "
            f"para recuperarse de una caída de mercado, aunque su actitud psicológica tolere más riesgo."
        )
    return tier, PORTFOLIO_MODELS[tier], age_note


# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "🎯 Perfil de riesgo", "🩺 Diagnóstico financiero", "💰 Meta de ahorro",
        "📋 Plan de acción", "🔀 Comparador", "📚 Glosario", "📄 Reporte",
    ]
)

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Cuestionario orientativo de actitud ante el riesgo")
    st.markdown(
        "<div class='small'>Leé cada afirmación en voz alta junto a tu cliente y marcá qué tan de acuerdo "
        "está con ella. No hay respuestas correctas o incorrectas: la idea es capturar su actitud real "
        "frente al riesgo, no lo que \"debería\" responder. Este cuestionario es orientativo — no es un "
        "instrumento psicométricamente validado ni sustituye una evaluación de capacidad de riesgo "
        "(edad, estabilidad de ingresos, dependientes, fondo de emergencia).</div>",
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
        portfolio_tier, portfolio_categories, portfolio_age_note = None, [], None
        st.info(f"🕓 {answered_count}/{len(RISK_QUESTIONS)} preguntas respondidas — {risk_desc}")
    else:
        risk_score = int(sum(answers))
        risk_category, risk_desc, alloc_fixed, alloc_variable = classify_risk(risk_score)
        portfolio_tier, portfolio_categories, portfolio_age_note = suggest_portfolio(risk_category, age)
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

    if portfolio_tier:
        st.markdown("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Cartera modelo sugerida")
        st.caption(
            "Según el perfil de riesgo del test y la edad del cliente, por categoría de instrumento "
            "(no recomienda títulos, emisores ni productos específicos)."
        )
        st.markdown(f"**Nivel sugerido: {portfolio_tier}**")
        if portfolio_age_note:
            st.info(f"ℹ️ {portfolio_age_note}")
        for categoria, pct in portfolio_categories:
            st.write(f"{categoria} — **{pct}%**")
            st.progress(pct / 100)
        st.caption(rep.PORTFOLIO_DISCLAIMER)
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
    # safe_div ya devuelve 0.0 si el denominador es 0; total_expenses ya está
    # calculado arriba, no hace falta re-sumar fixed+variable+debt_payment.
    emergency_months = safe_div(emergency_fund, total_expenses)

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
        help="Cuotas de deuda como % del ingreso NETO (no bruto, a diferencia del DTI que suelen usar los "
        "bancos). Es una referencia propia de esta herramienta, no un estándar regulado.",
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
    # Los indicadores "na" (dato no calculable, ej. sin ingreso) se excluyen del
    # denominador en vez de contar como 0 puntos: un dato faltante no es lo
    # mismo que un semáforo en rojo, y no debería bajar el puntaje máximo
    # posible de forma injusta.
    _rated = [r for r in diagnostic_ratings if r.key != "na"]
    health_max = len(_rated) * 2 if _rated else 8
    health_points = sum(2 if r.key == "good" else (1 if r.key == "warn" else 0) for r in _rated)
    has_critical = any(r.key == "bad" for r in diagnostic_ratings)
    if health_points <= health_max * 0.375:
        health_label = "Situación crítica"
    elif health_points <= health_max * 0.625 or has_critical:
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
    st.markdown("")
    with st.expander("📊 Regla 50/30/20 — ¿cómo se compara el cliente?"):
        st.caption(
            "La regla 50/30/20 sugiere destinar ~50% del ingreso a necesidades (gastos fijos + deudas), "
            "~30% a deseos (gastos variables), y ~20% a ahorro/inversión. Es una referencia general, no "
            "una meta exacta para todos los casos."
        )
        if income > 0:
            nec_pct = safe_div(fixed_expenses + debt_payment, income)
            deseo_pct = safe_div(variable_expenses, income)
            ahorro_pct = safe_div(savings_monthly, income)

            def pct_pill(actual, target, label_over, label_ok):
                if actual > target * 1.10:
                    return f"🔴 {label_over} ({actual*100:.0f}% vs ~{target*100:.0f}%)"
                if actual > target:
                    return f"🟡 Ajustado ({actual*100:.0f}% vs ~{target*100:.0f}%)"
                return f"🟢 {label_ok} ({actual*100:.0f}% vs ~{target*100:.0f}%)"

            c5020_1, c5020_2, c5020_3 = st.columns(3)
            c5020_1.metric("Necesidades", f"{nec_pct*100:.0f}%", help="Gastos fijos + cuota de deudas")
            c5020_1.caption(pct_pill(nec_pct, 0.50, "Por encima del 50%", "Dentro del 50%"))

            c5020_2.metric("Deseos", f"{deseo_pct*100:.0f}%", help="Gastos variables")
            c5020_2.caption(pct_pill(deseo_pct, 0.30, "Por encima del 30%", "Dentro del 30%"))

            c5020_3.metric("Ahorro", f"{ahorro_pct*100:.0f}%", help="Ahorro/inversión mensual declarado")
            if ahorro_pct >= 0.20:
                c5020_3.caption(f"🟢 Excelente ({ahorro_pct*100:.0f}% vs ~20%)")
            elif ahorro_pct >= 0.10:
                c5020_3.caption(f"🟡 En desarrollo ({ahorro_pct*100:.0f}% vs ~20%)")
            else:
                c5020_3.caption(f"🔴 Por debajo del 20% ({ahorro_pct*100:.0f}%)")

            total_asignado = nec_pct + deseo_pct + ahorro_pct
            if abs(total_asignado - 1.0) > 0.02:
                st.info(
                    f"ℹ️ La suma de las tres categorías ({total_asignado*100:.0f}%) no da 100% porque los "
                    "montos cargados no cubren todo el ingreso (queda dinero sin asignar) o lo superan."
                )
        else:
            st.caption("Completá el ingreso mensual para ver la comparativa 50/30/20.")

    st.markdown("")
    with st.expander("💳 Inventario de deudas (opcional) — para decidir cuál pagar primero"):
        st.caption(
            "Cargá cada deuda por separado (tarjetas, préstamos) para que la herramienta sugiera un orden "
            "de pago según la tasa de interés. Es opcional: si no lo completás, el plan de acción usa solo "
            "la 'Cuota mensual total de deudas' de arriba."
        )
        debts_df = st.data_editor(
            pd.DataFrame({
                "Deuda": pd.Series([], dtype="str"),
                "Saldo": pd.Series([], dtype="float"),
                "Tasa anual (%)": pd.Series([], dtype="float"),
                "Pago mínimo": pd.Series([], dtype="float"),
            }),
            num_rows="dynamic",
            column_config={
                "Deuda": st.column_config.TextColumn(required=True),
                "Saldo": st.column_config.NumberColumn(min_value=0.0, step=50_000.0, format="%.0f"),
                "Tasa anual (%)": st.column_config.NumberColumn(min_value=0.0, max_value=500.0, step=0.5, format="%.1f"),
                "Pago mínimo": st.column_config.NumberColumn(min_value=0.0, step=10_000.0, format="%.0f"),
            },
            key="fp_debts_editor",
            width="stretch",
        )
        debts_list = [
            rep.DebtItem(
                nombre=safe_cell_text(row["Deuda"]),
                saldo=safe_cell_num(row["Saldo"]),
                tasa_anual=safe_cell_num(row["Tasa anual (%)"]) / 100.0,
                pago_minimo=safe_cell_num(row["Pago mínimo"]),
            )
            for _, row in debts_df.iterrows()
            if safe_cell_text(row["Deuda"])
        ]
        debt_priority_note = None
        if debts_list:
            peor = max(debts_list, key=lambda d: d.tasa_anual)
            suma_pagos = sum(d.pago_minimo for d in debts_list)
            if abs(suma_pagos - debt_payment) > max(10_000.0, debt_payment * 0.05):
                st.info(
                    f"ℹ️ La suma de pagos mínimos del inventario ({fmt_pyg(suma_pagos)}) no coincide con la "
                    f"'Cuota mensual total de deudas' declarada arriba ({fmt_pyg(debt_payment)}). Ajustá el "
                    "que corresponda para que el diagnóstico sea consistente."
                )
            if peor.tasa_anual > 0.15 and emergency_rating.key != "bad":
                debt_priority_note = (
                    f"Con el fondo de emergencia en un nivel mínimo aceptable, priorizá cancelar "
                    f"'{peor.nombre}' (tasa {fmt_pct(peor.tasa_anual)} anual) antes que seguir acumulando "
                    f"más de 3-6 meses de colchón: la tasa de esa deuda probablemente supera cualquier "
                    f"rendimiento de mantener ese dinero ahorrado."
                )
            elif peor.tasa_anual > 0.15:
                debt_priority_note = (
                    f"Primero asegurá un mínimo de 3 meses de fondo de emergencia; recién después atacá la "
                    f"deuda de mayor tasa ('{peor.nombre}', {fmt_pct(peor.tasa_anual)} anual)."
                )
            else:
                debt_priority_note = (
                    f"Ninguna deuda cargada tiene una tasa claramente alta (la mayor es '{peor.nombre}', "
                    f"{fmt_pct(peor.tasa_anual)} anual); no hay una urgencia especial de pago acelerado por "
                    "sobre el plan general."
                )
            st.write(f"📌 {debt_priority_note}")
        else:
            debt_priority_note = None

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Meta de ahorro con monto y plazo")
    st.caption(
        "Opcional: definí junto al cliente una meta concreta (ej. un viaje, una cuota inicial) para "
        "traducir el diagnóstico en una acción mensual verificable."
    )
    goal_name = st.text_input("Nombre de la meta", "", placeholder="Ej: Fondo para cuota inicial de vivienda")
    colg1, colg2 = st.columns(2)
    with colg1:
        goal_amount = st.number_input("Monto objetivo (Gs.)", min_value=0.0, value=0.0, step=100_000.0)
    with colg2:
        goal_months = st.number_input("Plazo (meses)", min_value=0, value=0, step=1)
    st.markdown("</div>", unsafe_allow_html=True)

    has_goal = bool(goal_name.strip()) and goal_amount > 0 and goal_months > 0
    if has_goal:
        goal_required_monthly = goal_amount / goal_months
        goal_feasible = goal_required_monthly <= savings_monthly
        st.markdown("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Ahorro mensual necesario", fmt_pyg(goal_required_monthly))
        c2.metric("Ahorro mensual actual declarado", fmt_pyg(savings_monthly))
        if goal_feasible:
            st.success(f"✅ Alcanzable: con el ahorro actual, '{goal_name}' se cubre en {goal_months} meses o menos.")
        else:
            faltante = goal_required_monthly - savings_monthly
            st.warning(
                f"⚠️ No alcanza con el ahorro actual: falta destinar {fmt_pyg(faltante)} más por mes a esta "
                f"meta, o extender el plazo. A este ritmo actual ({fmt_pyg(savings_monthly)}/mes), la meta "
                f"tomaría {(goal_amount / savings_monthly):.0f} meses en vez de {goal_months}."
                if savings_monthly > 0 else
                "⚠️ No alcanza: el cliente no declaró ahorro mensual actual, así que no puede cubrir esta "
                "meta sin antes generar un excedente."
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        goal_required_monthly = None
        goal_feasible = None
        st.caption("Completá nombre, monto y plazo (mayores a 0) para calcular la meta.")

with tab4:
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

    # Conecta el inventario de deudas (pestaña Diagnóstico) con el orden del plan,
    # en vez de asumir siempre "primero fondo de emergencia completo, después deuda".
    if debts_list and debt_priority_note:
        action_plan.append(f"💳 Deudas: {debt_priority_note}")

    # Conecta la meta de ahorro (si se cargó) con el plan.
    if has_goal:
        if goal_feasible:
            action_plan.append(
                f"🏁 Meta '{goal_name}': alcanzable con el ahorro actual en {goal_months} meses; dale "
                f"seguimiento mes a mes junto al cliente."
            )
        else:
            action_plan.append(
                f"🏁 Meta '{goal_name}': con el ahorro actual no se llega en el plazo definido; ajustá el "
                f"monto mensual destinado o el plazo (ver pestaña 'Meta de ahorro')."
            )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Plan de acción sugerido")
    for item in action_plan:
        st.write(item)
    st.markdown("</div>", unsafe_allow_html=True)

    st.info(rep.DISCLAIMER)

    st.markdown("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Compromisos acordados con el cliente")
    st.caption(
        "Convertí las recomendaciones de arriba en compromisos concretos: qué se hace, para cuándo y quién "
        "es responsable. Podés editar el texto, agregar o borrar filas."
    )
    _suggested = [strip_emoji_prefix(item) for item in action_plan[:3]]
    commitments_df = st.data_editor(
        pd.DataFrame({
            "Compromiso": _suggested,
            "Fecha": [None] * len(_suggested),
            "Responsable": ["Cliente"] * len(_suggested),
        }),
        num_rows="dynamic",
        column_config={
            "Compromiso": st.column_config.TextColumn(required=True, width="large"),
            "Fecha": st.column_config.DateColumn(),
            "Responsable": st.column_config.SelectboxColumn(options=["Cliente", "Consultor/a", "Ambos"]),
        },
        key="fp_commitments_editor",
        width="stretch",
    )
    commitments = []
    for _, row in commitments_df.iterrows():
        texto = safe_cell_text(row["Compromiso"])
        if not texto:
            continue
        fecha_val = row["Fecha"]
        fecha_str = fecha_val.isoformat() if pd.notna(fecha_val) and hasattr(fecha_val, "isoformat") else ""
        commitments.append((texto, fecha_str, safe_cell_text(row["Responsable"])))
    st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Comparador antes / después")
    st.caption(
        "Simulá con el cliente el efecto de reducir un gasto o aumentar el ahorro. Es solo una vista "
        "interactiva para la conversación: NO cambia los datos de la pestaña Diagnóstico ni el reporte final."
    )
    colw1, colw2 = st.columns(2)
    with colw1:
        reduce_variable = st.number_input(
            "Reducir gastos variables en (Gs./mes)", min_value=0.0, max_value=float(variable_expenses),
            value=0.0, step=50_000.0,
        )
    with colw2:
        increase_savings = st.number_input(
            "Aumentar ahorro/inversión en (Gs./mes)", min_value=0.0, value=0.0, step=50_000.0,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    colB, colA = st.columns(2)
    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Antes")
        st.metric("Excedente antes de ahorro", fmt_pyg(cashflow))
        st.metric("Tasa de ahorro", fmt_pct(savings_rate) if savings_rating.key != "na" else "—")
        st.metric("Disponible después de ahorro", fmt_pyg(cashflow - savings_monthly))
        st.markdown("</div>", unsafe_allow_html=True)
    with colA:
        new_variable_expenses = variable_expenses - reduce_variable
        new_savings_monthly = savings_monthly + increase_savings
        new_total_expenses = fixed_expenses + new_variable_expenses + debt_payment
        new_cashflow = income - new_total_expenses
        new_savings_rate = safe_div(new_savings_monthly, income)
        new_savings_rating = rate_savings(new_savings_rate, income)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Después")
        st.metric(
            "Excedente antes de ahorro", fmt_pyg(new_cashflow),
            delta=fmt_pyg(new_cashflow - cashflow),
        )
        st.metric(
            "Tasa de ahorro",
            fmt_pct(new_savings_rate) if new_savings_rating.key != "na" else "—",
            delta=(fmt_pct(new_savings_rate - savings_rate) if new_savings_rating.key != "na" and savings_rating.key != "na" else None),
        )
        st.metric(
            "Disponible después de ahorro", fmt_pyg(new_cashflow - new_savings_monthly),
            delta=fmt_pyg((new_cashflow - new_savings_monthly) - (cashflow - savings_monthly)),
        )
        st.markdown("</div>", unsafe_allow_html=True)

with tab6:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Glosario para explicarle al cliente")
    st.caption("Definiciones breves, en lenguaje simple, para usar durante la consultoría.")
    glosario = [
        ("Excedente antes de ahorro", "Lo que queda del ingreso después de pagar gastos y cuotas de deuda, SIN restar lo que se ahorra. Si es negativo, se está gastando más de lo que se gana."),
        ("Disponible después de ahorro", "El excedente menos lo que el cliente dice que ahorra. Si da negativo, el ahorro declarado no es sostenible con el ingreso actual."),
        ("Tasa de ahorro", "Qué porcentaje del ingreso se destina a ahorro/inversión cada mes."),
        ("Endeudamiento (DTI)", "Qué porcentaje del ingreso neto se va en cuotas de deuda. A mayor DTI, menos margen para imprevistos."),
        ("Fondo de emergencia", "Ahorro líquido para cubrir gastos si el cliente pierde su ingreso. Se mide en 'meses de gastos cubiertos'."),
        ("Perfil de riesgo (tolerancia)", "Qué tan cómodo se siente el cliente asumiendo pérdidas temporales a cambio de mayor retorno. Es una actitud psicológica, no una medida de cuánto riesgo puede permitirse."),
        ("Capacidad de riesgo", "Cuánta pérdida puede absorber el cliente en la práctica, según su edad, estabilidad de ingresos, dependientes y colchón de emergencia. Es distinta de la tolerancia (arriba) y esta herramienta no la calcula por separado."),
        ("Patrimonio neto", "Todo lo que el cliente posee (activos) menos todo lo que debe (pasivos). Puede ser 0 sin que sea un error de carga."),
        ("Regla 50/30/20", "Guía general de presupuesto: ~50% del ingreso a necesidades, ~30% a deseos, ~20% a ahorro o pago extra de deuda. No es una meta obligatoria, es un punto de referencia."),
        ("Mezcla ilustrativa", "Un ejemplo de cómo repartir ahorros entre instrumentos de bajo y mayor riesgo según el perfil del test. No es una recomendación de inversión concreta."),
    ]
    for termino, definicion in glosario:
        with st.expander(termino):
            st.write(definicion)
    st.markdown("</div>", unsafe_allow_html=True)

with tab7:
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
            goal_name if has_goal else None, goal_amount if has_goal else None, goal_months if has_goal else None,
            tuple((d.nombre, d.saldo, d.tasa_anual, d.pago_minimo) for d in debts_list),
            tuple(commitments),
            portfolio_tier, tuple(portfolio_categories),
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
                savings_goal_name=goal_name if has_goal else None,
                savings_goal_amount=float(goal_amount) if has_goal else None,
                savings_goal_months=int(goal_months) if has_goal else None,
                savings_goal_required_monthly=float(goal_required_monthly) if has_goal else None,
                savings_goal_feasible=bool(goal_feasible) if has_goal else None,
                debts=debts_list,
                debt_priority_note=debt_priority_note,
                commitments=commitments,
                portfolio_tier=portfolio_tier,
                portfolio_categories=portfolio_categories,
                portfolio_age_note=portfolio_age_note,
            )
            # Recién acá se calculan el TXT, el PDF y el Word — no en cada rerun del script.
            st.session_state["fp_report_txt"] = rep.build_text_report(report)
            st.session_state["fp_report_pdf"] = (
                rep.generate_pdf(report) if rep.REPORTLAB_OK else None
            )
            st.session_state["fp_report_docx"] = (
                rep.generate_docx(report) if rep.DOCX_OK else None
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
            if st.session_state.get("fp_report_docx") is not None:
                st.download_button(
                    "⬇️ Descargar reporte (Word)",
                    data=st.session_state["fp_report_docx"],
                    file_name="diagnostico_finanzas_personales.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            elif not rep.DOCX_OK:
                st.info("Para exportar Word, agrega `python-docx` a requirements.txt.")
        else:
            st.info(
                "Presioná \"🔄 Generar reporte con los datos actuales\" para habilitar la descarga."
            )

# Aviso temprano (visible desde cualquier pestaña) si todavía no se tocaron
# los valores de ejemplo precargados — complementa el checkbox de
# confirmación de la pestaña Reporte, que recién se ve al llegar ahí.
if not consultant.strip() and not client.strip() and income == 6_000_000.0:
    st.sidebar.warning(
        "⚠️ Los datos parecen ser los valores de ejemplo precargados. Cargá los datos reales del "
        "cliente antes de generar el reporte."
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
