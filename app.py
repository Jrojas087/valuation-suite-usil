import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import xlsxwriter
import json
from datetime import datetime

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Valuation Master Suite - USIL",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    .footer {position: fixed; bottom: 0; left: 0; width: 100%; background-color: transparent; color: #888; text-align: center; padding: 10px; font-size: 12px; z-index: 999;}
    .audit-box {background-color: #e8f4f8; border-left: 5px solid #002060; padding: 10px; margin: 10px 0; border-radius: 4px; font-size: 0.9em;}
    .red-flag {background-color: #fde8e8; border: 1px solid #f98080; color: #c53030; padding: 10px; border-radius: 5px; margin-bottom: 10px;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. GESTIÓN DE ESTADO (PERSISTENCIA) ---
def load_state(uploaded_file):
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            for key, value in data.items():
                st.session_state[key] = value
            st.success("¡Caso cargado exitosamente!")
            st.rerun()
        except Exception as e:
            st.error(f"Error al cargar archivo: {e}")

def get_current_inputs():
    # Helper para recolectar inputs actuales para guardar o calcular
    return {
        'sales_t0': st.session_state.get('sales_t0', 10000000.0),
        'cogs_t0': st.session_state.get('cogs_t0', 6000000.0),
        'opex_t0': st.session_state.get('opex_t0', 1500000.0),
        'deprec_t0': st.session_state.get('deprec_t0', 400000.0),
        'tax_rate': st.session_state.get('tax_rate', 25),
        'ar_t0': st.session_state.get('ar_t0', 830000.0),
        'inv_t0': st.session_state.get('inv_t0', 750000.0),
        'ap_t0': st.session_state.get('ap_t0', 600000.0),
        'debt_t0': st.session_state.get('debt_t0', 2000000.0),
        'equity_book_t0': st.session_state.get('equity_book_t0', 3000000.0),
        'g_sales': st.session_state.get('g_sales', 5.0),
        'g_term': st.session_state.get('g_term', 2.5),
        'capex': st.session_state.get('capex', 4.5),
        'de_target': st.session_state.get('de_target', 0.40),
        'debt_amort': st.session_state.get('debt_amort', 200000.0),
        'debt_new': st.session_state.get('debt_new', 50000.0),
        'rf': st.session_state.get('rf', 4.0),
        'erp': st.session_state.get('erp', 5.5),
        'beta_u': st.session_state.get('beta_u', 0.90),
        'kd': st.session_state.get('kd', 7.5),
        # Inputs Valoración Relativa
        'peer_ev_ebitda': st.session_state.get('peer_ev_ebitda', 8.0),
        'peer_pe': st.session_state.get('peer_pe', 12.0)
    }

# --- 3. LÓGICA DE NEGOCIO Y AUDITORÍA ---
def check_red_flags(inputs):
    """MEJORA 5: Motor de Detección de Inconsistencias (Red Flags)"""
    flags = []
    
    # 1. Crecimiento Perpetuo vs Economía
    if inputs['g_term']/100 > 0.04:
        flags.append("⚠️ **Crecimiento Irreal:** El crecimiento a perpetuidad (g) es > 4%. Es improbable que una empresa crezca más que la economía mundial para siempre.")
    
    # 2. Descapitalización (Capex vs Depreciación)
    sales_proj = inputs['sales_t0'] * (1 + inputs['g_sales']/100)
    capex_abs = sales_proj * (inputs['capex']/100)
    deprec_rel = inputs['deprec_t0'] # Aprox
    if capex_abs < deprec_rel * 0.8:
        flags.append("⚠️ **Posible Descapitalización:** El Capex proyectado es menor que la Depreciación histórica. La empresa podría estar 'comiéndose' sus activos.")
    
    # 3. WACC vs ROIC (Simplificado)
    # Si el margen es muy bajo y el WACC alto, destruye valor.
    margin = (inputs['sales_t0'] - inputs['cogs_t0'] - inputs['opex_t0']) / inputs['sales_t0']
    if margin < 0.05:
        flags.append("⚠️ **Margen Crítico:** El margen EBITDA es menor al 5%. Revisa si el modelo de negocio es viable.")

    return flags

def calculate_dcf(inputs):
    # Desempaquetar
    tax = inputs['tax_rate'] / 100
    rf, erp, kd = inputs['rf']/100, inputs['erp']/100, inputs['kd']/100
    g_sales, g_term = inputs['g_sales']/100, inputs['g_term']/100
    capex_pct = inputs['capex']/100
    
    # WACC
    beta_l = inputs['beta_u'] * (1 + (1-tax)*inputs['de_target'])
    ke = rf + (beta_l * erp)
    kd_net = kd * (1 - tax)
    wd = inputs['de_target'] / (1 + inputs['de_target'])
    we = 1 / (1 + inputs['de_target'])
    wacc = (we * ke) + (wd * kd_net)

    if wacc <= g_term:
        return {'error': f"⚠️ Error Matemático: WACC ({wacc:.1%}) <= Crecimiento g ({g_term:.1%})."}

    # Proyecciones
    schedule = []
    nwc_prev = inputs['ar_t0'] + inputs['inv_t0'] - inputs['ap_t0']
    curr_sales = inputs['sales_t0']
    curr_debt = inputs['debt_t0']
    
    margin_base = (inputs['sales_t0'] - inputs['cogs_t0'] - inputs['opex_t0']) / inputs['sales_t0']
    dso = (inputs['ar_t0'] / inputs['sales_t0']) * 360 if inputs['sales_t0'] > 0 else 0
    dio = (inputs['inv_t0'] / inputs['cogs_t0']) * 360 if inputs['cogs_t0'] > 0 else 0
    dpo = (inputs['ap_t0'] / inputs['cogs_t0']) * 360 if inputs['cogs_t0'] > 0 else 0
    
    schedule.append({"Año": 0, "Ventas": inputs['sales_t0'], "EBITDA": inputs['sales_t0']*margin_base, "FCFF": 0})
    fcff_list = []

    for i in range(1, 6):
        curr_sales *= (1 + g_sales)
        ebitda = curr_sales * margin_base
        deprec = curr_sales * (inputs['deprec_t0']/inputs['sales_t0']) if inputs['sales_t0'] > 0 else 0
        ebit = ebitda - deprec
        nopat = ebit * (1-tax)
        
        cogs_proj = curr_sales * (inputs['cogs_t0']/inputs['sales_t0']) if inputs['sales_t0'] > 0 else 0
        nwc_curr = (curr_sales/360)*dso + (cogs_proj/360)*dio - (cogs_proj/360)*dpo
        var_nwc = nwc_curr - nwc_prev
        nwc_prev = nwc_curr
        
        capex = curr_sales * capex_pct
        fcff = nopat + deprec - capex - var_nwc
        fcff_list.append(fcff)
        
        # Deuda dinámica para FCFE (simplificado para DCF principal)
        curr_debt = curr_debt - inputs['debt_amort'] + inputs['debt_new']
        
        schedule.append({
            "Año": i, "Ventas": curr_sales, "EBITDA": ebitda,
            "NOPAT": nopat, "Capex": capex, "Var NWC": var_nwc, "FCFF": fcff
        })

    tv = fcff_list[-1] * (1+g_term) / (wacc - g_term)
    vp_flows = sum([f / ((1+wacc)**(i+1)) for i, f in enumerate(fcff_list)])
    vp_tv = tv / ((1+wacc)**5)
    ev = vp_flows + vp_tv
    equity_val = ev - inputs['debt_t0']

    # MEJORA 4: Datos para Escenarios y Valoración Relativa
    # Calculamos métricas TTM (Trailing Twelve Months) o forward para múltiplos
    ebitda_t1 = schedule[1]['EBITDA']
    net_income_t1 = (schedule[1]['NOPAT'] - (inputs['debt_t0']*kd_net)) # Aprox rápida
    
    return {
        'wacc': wacc, 'ke': ke, 'kd_net': kd_net, 'beta_l': beta_l,
        'ev': ev, 'equity': equity_val, 'debt': inputs['debt_t0'],
        'vp_flows': vp_flows, 'vp_tv': vp_tv, 
        'df': pd.DataFrame(schedule),
        'g_term': g_term, 'fcff_last': fcff_list[-1], 'g_sales': g_sales,
        'inputs': inputs,
        'ebitda_t1': ebitda_t1, 'net_income_t1': net_income_t1
    }

# --- 4. SIDEBAR: CONTROL Y PERSISTENCIA ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Logo_USIL.png/640px-Logo_USIL.png", width=140)
    st.title("Panel de Control")
    st.markdown("**Prof. Jorge Rojas (MBA)**")
    
    # MEJORA 3: GUARDAR Y CARGAR CASO
    with st.expander("💾 Guardar / Cargar Caso", expanded=False):
        # Descargar
        current_data = get_current_inputs()
        json_str = json.dumps(current_data)
        st.download_button("📥 Bajar Caso (.json)", json_str, file_name=f"valuation_case_{datetime.now().strftime('%H%M')}.json", mime="application/json")
        
        # Cargar
        uploaded_file = st.file_uploader("📤 Subir Caso (.json)", type="json")
        if uploaded_file:
            if st.button("Restaurar Datos"):
                load_state(uploaded_file)
    
    # MEJORA 1: AUDIT MODE TOGGLE
    audit_mode = st.toggle("🎓 Audit Mode (Docente)", value=False, help="Activa explicaciones detalladas de fórmulas")
    
    st.divider()
    
    alumno = st.text_input("Analista / Alumno", value="Estudiante MBA")
    proyecto = st.text_input("Empresa Target", value="Empresa Modelo S.A.")
    
    with st.expander("⚙️ Tasas de Mercado (CAPM)", expanded=True):
        rf_in = st.number_input("Tasa Libre Riesgo (Rf %)", value=4.0, key='rf')
        erp_in = st.number_input("Prima Riesgo (ERP %)", value=5.5, key='erp')
        beta_u = st.number_input("Beta Desapalancado", value=0.90, key='beta_u')
        kd_in = st.number_input("Costo Deuda (Kd %)", value=7.5, key='kd')

# --- 5. INTERFAZ PRINCIPAL ---
st.title(f"💎 Valuation Suite: {proyecto}")

# TABS DE INPUTS
tab_in1, tab_in2, tab_in3 = st.tabs(["1️⃣ Estados Financieros", "2️⃣ Supuestos DCF", "3️⃣ Valoración Relativa"])

with tab_in1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("P&L (Estado de Resultados)")
        st.number_input("Ventas ($)", value=10000000.0, key='sales_t0')
        st.number_input("Costo Ventas ($)", value=6000000.0, key='cogs_t0')
        st.number_input("Gastos Op. ($)", value=1500000.0, key='opex_t0')
        st.number_input("Depreciación ($)", value=400000.0, key='deprec_t0')
        st.slider("Tax Corporativo (%)", 0, 40, 25, key='tax_rate')
    with c2:
        st.subheader("Balance General")
        st.number_input("Ctas Cobrar", value=830000.0, key='ar_t0')
        st.number_input("Inventarios", value=750000.0, key='inv_t0')
        st.number_input("Ctas Pagar", value=600000.0, key='ap_t0')
        st.number_input("Deuda Financiera Total", value=2000000.0, key='debt_t0')
        st.number_input("Patrimonio Contable", value=3000000.0, key='equity_book_t0')

with tab_in2:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Crecimiento**")
        st.number_input("Crec. Ventas (%)", value=5.0, key='g_sales')
        st.number_input("Crec. Perpetuo (g) (%)", value=2.5, key='g_term')
        st.number_input("Capex (% Ventas)", value=4.5, key='capex')
    with c2:
        st.markdown("**Estructura**")
        st.number_input("Target D/E", value=0.40, key='de_target')
    with c3:
        st.markdown("**Deuda**")
        st.number_input("Amort. Deuda ($)", value=200000.0, key='debt_amort')
        st.number_input("Nueva Deuda ($)", value=50000.0, key='debt_new')

with tab_in3:
    st.info("💡 Ingrese múltiplos de empresas comparables para triangular el valor.")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Múltiplo EV/EBITDA Industria (x)", value=8.0, step=0.5, key='peer_ev_ebitda')
    with c2:
        st.number_input("Múltiplo Price/Earnings (P/E) Industria (x)", value=12.0, step=0.5, key='peer_pe')

if st.button("🚀 EJECUTAR AUDITORÍA Y VALORACIÓN", type="primary", use_container_width=True):
    inputs = get_current_inputs()
    results = calculate_dcf(inputs)
    
    if 'error' in results:
        st.error(results['error'])
    else:
        st.session_state['res'] = results
        # MEJORA 5: Limpiar simulaciones anteriores
        if 'sim_data' in st.session_state: del st.session_state['sim_data']

# --- 6. RESULTADOS ---
if 'res' in st.session_state:
    res = st.session_state['res']
    inputs = res['inputs']
    st.divider()
    
    # MEJORA 2: RED FLAGS (AUDITORÍA)
    flags = check_red_flags(inputs)
    if flags:
        with st.expander(f"🚩 ALERTA DE AUDITORÍA: {len(flags)} Hallazgos Detectados", expanded=True):
            for flag in flags:
                st.markdown(f"<div class='red-flag'>{flag}</div>", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["📊 Dashboard", "⚖️ Valoración Relativa", "🔮 Escenarios", "🌪️ Riesgo", "📄 Reporte"])
    
    # --- DASHBOARD ---
    with t1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Enterprise Value", f"${res['ev']/1e6:,.1f} M")
        c2.metric("Deuda Neta", f"${res['debt']/1e6:,.1f} M")
        c3.metric("Equity Value", f"${res['equity']/1e6:,.1f} M", delta="DCF Intrínseco")
        
        # MEJORA 1: AUDIT MODE (WACC)
        if audit_mode:
            st.markdown(f"""
            <div class='audit-box'>
            <b>🕵️‍♂️ Auditoría del WACC ({res['wacc']:.2%}):</b><br>
            Calculado como: <code>(Ke * We) + (Kd_net * Wd)</code><br>
            • Costo Equity (Ke): {res['ke']:.2%} [Rf + Beta * ERP]<br>
            • Costo Deuda Neto: {res['kd_net']:.2%} [Kd * (1-t)]
            </div>
            """, unsafe_allow_html=True)

        fig = go.Figure(go.Waterfall(
            orientation = "v",
            measure = ["relative", "relative", "total", "relative", "total"],
            x = ["VP Flujos", "VP Terminal", "EV", "(-) Deuda", "Equity"],
            text = [f"{res['vp_flows']/1e6:.1f}", f"{res['vp_tv']/1e6:.1f}", f"{res['ev']/1e6:.1f}", f"-{res['debt']/1e6:.1f}", f"{res['equity']/1e6:.1f}"],
            y = [res['vp_flows'], res['vp_tv'], 0, -res['debt'], 0],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig.update_layout(template="streamlit", height=300, title="Puente de Valoración (M$)")
        st.plotly_chart(fig, use_container_width=True)

    # --- VALORACIÓN RELATIVA (MEJORA 4) ---
    with t2:
        st.subheader("Triangulación de Valor (Benchmarking)")
        
        # Cálculos Múltiplos
        val_ev_ebitda = inputs['peer_ev_ebitda'] * res['ebitda_t1']
        equity_ev_ebitda = val_ev_ebitda - inputs['debt_t0']
        
        val_pe = inputs['peer_pe'] * res['net_income_t1'] if res['net_income_t1'] > 0 else 0
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.markdown("#### Según EV/EBITDA")
            st.metric("Valor Implícito (Equity)", f"${equity_ev_ebitda/1e6:,.1f} M", 
                      delta=f"{((equity_ev_ebitda/res['equity'])-1):.1%} vs DCF")
            if audit_mode:
                st.caption(f"Cálculo: EBITDA Proyectado (${res['ebitda_t1']/1e6:.1f}M) x Múltiplo ({inputs['peer_ev_ebitda']}x) - Deuda")
                
        with col_r2:
            st.markdown("#### Según Price/Earnings")
            st.metric("Valor Implícito (Equity)", f"${val_pe/1e6:,.1f} M",
                      delta=f"{((val_pe/res['equity'])-1):.1%} vs DCF")
            if audit_mode:
                 st.caption(f"Cálculo: Utilidad Neta Proyectada (${res['net_income_t1']/1e6:.1f}M) x Múltiplo ({inputs['peer_pe']}x)")

        # Gráfico Comparativo
        comp_df = pd.DataFrame({
            'Método': ['DCF (Intrínseco)', 'Múltiplo EV/EBITDA', 'Múltiplo P/E'],
            'Valor Equity ($M)': [res['equity']/1e6, equity_ev_ebitda/1e6, val_pe/1e6]
        })
        fig_comp = px.bar(comp_df, x='Método', y='Valor Equity ($M)', color='Método', title="Rango de Valoración")
        st.plotly_chart(fig_comp, use_container_width=True)

    # --- ESCENARIOS (MEJORA 5) ---
    with t3:
        st.subheader("Análisis de Escenarios Discretos")
        st.caption("Comparativa automática variando Crecimiento y WACC.")
        
        # Definir escenarios
        scenarios = {
            "Pesimista": {'g': inputs['g_sales']*0.8, 'wacc_add': 0.01},
            "Base": {'g': inputs['g_sales'], 'wacc_add': 0.0},
            "Optimista": {'g': inputs['g_sales']*1.2, 'wacc_add': -0.01}
        }
        
        res_scenarios = []
        for name, params in scenarios.items():
            # Clonar inputs y modificar
            temp_inputs = inputs.copy()
            temp_inputs['g_sales'] = params['g']
            temp_inputs['g_term'] = params['g'] * 0.5 # Aprox g_term sigue a g_sales
            # Recalcular WACC simulado (ajustando ERP en el input clonado para efecto)
            temp_inputs['erp'] = inputs['erp'] + (params['wacc_add']*100) 
            
            try:
                calc = calculate_dcf(temp_inputs)
                if 'error' not in calc:
                    res_scenarios.append([name, calc['equity'], calc['ev']])
            except:
                pass
        
        df_scen = pd.DataFrame(res_scenarios, columns=["Escenario", "Equity Value", "Enterprise Value"])
        df_scen["Equity ($M)"] = df_scen["Equity Value"] / 1e6
        
        st.table(df_scen[["Escenario", "Equity ($M)"]].style.format({"Equity ($M)": "${:,.1f}"}))
        
        if audit_mode:
            st.info("Nota: El escenario Pesimista reduce crecimiento un 20% y aumenta riesgo (ERP) un 1%. El Optimista hace lo inverso.")

    # --- RIESGO (MONTECARLO) ---
    with t4:
        st.subheader("Simulación Montecarlo")
        if st.button("🎲 Correr 5,000 Escenarios"):
            sim_wacc = np.random.normal(res['wacc'], 0.015, 5000)
            sim_g = np.random.normal(res['g_term'], 0.005, 5000)
            sim_wacc = np.maximum(sim_wacc, sim_g + 0.005)
            tv_s = res['fcff_last']*(1+sim_g)/(sim_wacc-sim_g)
            ev_s = (tv_s / ((1+sim_wacc)**5)) + res['vp_flows']
            st.session_state['sim_data'] = ev_s - res['debt']

        if 'sim_data' in st.session_state:
            data = st.session_state['sim_data']
            var_5 = np.percentile(data, 5)
            fig_hist = px.histogram(x=data/1e6, nbins=40, title="Distribución de Valor (Equity)")
            fig_hist.add_vline(x=np.mean(data)/1e6, line_color="green", annotation_text="Media")
            fig_hist.add_vline(x=var_5/1e6, line_color="red", annotation_text="VaR 5%")
            st.plotly_chart(fig_hist, use_container_width=True)
            st.metric("Riesgo de Caída (VaR 5%)", f"${var_5/1e6:,.1f} M")

    # --- EXCEL ---
    with t5:
        st.subheader("Reporte Auditoría (Excel)")
        if st.button("📥 Descargar Reporte Completo"):
            output = io.BytesIO()
            workbook = xlsxwriter.Workbook(output, {'in_memory': True, 'nan_inf_to_errors': True})
            
            # Formatos
            fmt_head = workbook.add_format({'bold': True, 'bg_color': '#002060', 'font_color': 'white', 'border': 1})
            fmt_curr = workbook.add_format({'num_format': '$ #,##0', 'border': 1})
            
            # Hoja 1: Resumen
            ws = workbook.add_worksheet("Resumen")
            ws.write('A1', f"VALORACIÓN: {proyecto}", fmt_head)
            ws.write('A3', "MÉTRICAS", fmt_head)
            ws.write('A4', "DCF Equity")
            ws.write('B4', res['equity'], fmt_curr)
            ws.write('A5', "Valor Relativo (EV/EBITDA)")
            ws.write('B5', equity_ev_ebitda, fmt_curr)
            
            if flags:
                ws.write('A8', "RED FLAGS DETECTADAS", fmt_head)
                for i, f in enumerate(flags):
                    ws.write(8+i, 0, f)
            
            workbook.close()
            output.seek(0)
            st.download_button("Bajar .xlsx", data=output, file_name=f"Audit_Valuation_{proyecto}.xlsx")

st.markdown("<div class='footer'>Valuation Master Suite v4.0 (Audit Edition) © 2026 - USIL</div>", unsafe_allow_html=True)
