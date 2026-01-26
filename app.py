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
    .red-flag {background-color: #fde8e8; border: 1px solid #f98080; color: #c53030; padding: 10px; border-radius: 5px; margin-bottom: 10px;}
    .scenario-table {font-size: 0.9em;}
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
        'peer_ev_ebitda': st.session_state.get('peer_ev_ebitda', 8.0),
        'peer_pe': st.session_state.get('peer_pe', 12.0)
    }

# --- 3. LÓGICA DE NEGOCIO ---
def check_red_flags(inputs):
    flags = []
    if inputs['g_term']/100 > 0.04:
        flags.append("⚠️ **Crecimiento Irreal:** g > 4% (Mayor al PBI mundial).")
    
    sales_proj = inputs['sales_t0'] * (1 + inputs['g_sales']/100)
    capex_abs = sales_proj * (inputs['capex']/100)
    if capex_abs < inputs['deprec_t0'] * 0.8:
        flags.append("⚠️ **Descapitalización:** Capex proyectado < Depreciación histórica.")
    
    margin = (inputs['sales_t0'] - inputs['cogs_t0'] - inputs['opex_t0']) / inputs['sales_t0']
    if margin < 0.05:
        flags.append("⚠️ **Margen Crítico:** EBITDA < 5%. Riesgo operativo alto.")
    return flags

def calculate_dcf(inputs):
    tax = inputs['tax_rate'] / 100
    rf, erp, kd = inputs['rf']/100, inputs['erp']/100, inputs['kd']/100
    g_sales, g_term = inputs['g_sales']/100, inputs['g_term']/100
    capex_pct = inputs['capex']/100
    
    beta_l = inputs['beta_u'] * (1 + (1-tax)*inputs['de_target'])
    ke = rf + (beta_l * erp)
    kd_net = kd * (1 - tax)
    wd = inputs['de_target'] / (1 + inputs['de_target'])
    we = 1 / (1 + inputs['de_target'])
    wacc = (we * ke) + (wd * kd_net)

    if wacc <= g_term:
        return {'error': f"⚠️ Error: WACC ({wacc:.1%}) <= g ({g_term:.1%})."}

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

    ebitda_t1 = schedule[1]['EBITDA']
    net_income_t1 = (schedule[1]['NOPAT'] - (inputs['debt_t0']*kd_net))
    
    return {
        'wacc': wacc, 'ke': ke, 'kd_net': kd_net,
        'ev': ev, 'equity': equity_val, 'debt': inputs['debt_t0'],
        'vp_flows': vp_flows, 'vp_tv': vp_tv, 
        'df': pd.DataFrame(schedule),
        'g_term': g_term, 'fcff_last': fcff_list[-1], 'g_sales': g_sales,
        'inputs': inputs,
        'ebitda_t1': ebitda_t1, 'net_income_t1': net_income_t1
    }

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Logo_USIL.png/640px-Logo_USIL.png", width=140)
    st.title("Panel de Control")
    st.markdown("**Prof. Jorge Rojas (MBA)**")
    
    with st.expander("💾 Guardar / Cargar Caso"):
        current_data = get_current_inputs()
        json_str = json.dumps(current_data)
        st.download_button("📥 Bajar Caso (.json)", json_str, file_name="valuation_case.json", mime="application/json")
        uploaded_file = st.file_uploader("📤 Subir Caso", type="json")
        if uploaded_file and st.button("Restaurar Datos"):
            load_state(uploaded_file)
    
    audit_mode = st.toggle("🎓 Modo Docente", value=False)
    st.divider()
    
    alumno = st.text_input("Analista / Alumno", value="Estudiante MBA")
    proyecto = st.text_input("Empresa Target", value="Empresa Modelo S.A.")
    
    with st.expander("⚙️ Tasas de Mercado (CAPM)", expanded=True):
        st.number_input("Rf (%)", value=4.0, key='rf')
        st.number_input("ERP (%)", value=5.5, key='erp')
        st.number_input("Beta U", value=0.90, key='beta_u')
        st.number_input("Kd (%)", value=7.5, key='kd')

# --- 5. INTERFAZ ---
st.title(f"💎 Valuation Suite: {proyecto}")
tab_in1, tab_in2, tab_in3 = st.tabs(["1️⃣ Financieros", "2️⃣ Supuestos", "3️⃣ Múltiplos"])

with tab_in1:
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Ventas ($)", value=10000000.0, key='sales_t0')
        st.number_input("Costo Ventas ($)", value=6000000.0, key='cogs_t0')
        st.number_input("Gastos Op. ($)", value=1500000.0, key='opex_t0')
        st.number_input("Depreciación ($)", value=400000.0, key='deprec_t0')
        st.slider("Tax (%)", 0, 40, 25, key='tax_rate')
    with c2:
        st.number_input("Ctas Cobrar", value=830000.0, key='ar_t0')
        st.number_input("Inventarios", value=750000.0, key='inv_t0')
        st.number_input("Ctas Pagar", value=600000.0, key='ap_t0')
        st.number_input("Deuda Total", value=2000000.0, key='debt_t0')
        st.number_input("Patrimonio", value=3000000.0, key='equity_book_t0')

with tab_in2:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Crec. Ventas (%)", value=5.0, key='g_sales')
        st.number_input("Crec. Perpetuo (%)", value=2.5, key='g_term')
        st.number_input("Capex (% Ventas)", value=4.5, key='capex')
    with c2:
        st.number_input("Target D/E", value=0.40, key='de_target')
    with c3:
        st.number_input("Amort. Deuda", value=200000.0, key='debt_amort')
        st.number_input("Nueva Deuda", value=50000.0, key='debt_new')

with tab_in3:
    c1, c2 = st.columns(2)
    c1.number_input("EV/EBITDA Industria (x)", value=8.0, key='peer_ev_ebitda')
    c2.number_input("P/E Industria (x)", value=12.0, key='peer_pe')

if st.button("🚀 EJECUTAR MODELO", type="primary", use_container_width=True):
    inputs = get_current_inputs()
    results = calculate_dcf(inputs)
    if 'error' in results:
        st.error(results['error'])
    else:
        st.session_state['res'] = results
        # Calcular Escenarios
        scenarios = []
        for name, g_fac, w_add in [("Pesimista", 0.8, 0.01), ("Base", 1.0, 0.0), ("Optimista", 1.2, -0.01)]:
            tmp = inputs.copy()
            tmp['g_sales'] *= g_fac
            tmp['g_term'] *= g_fac
            tmp['erp'] += w_add*100
            try: 
                calc = calculate_dcf(tmp)
                scenarios.append([name, calc['equity'], calc['ev']])
            except: pass
        st.session_state['scenarios'] = pd.DataFrame(scenarios, columns=["Escenario", "Equity Value", "EV"])

# --- 6. RESULTADOS ---
if 'res' in st.session_state:
    res = st.session_state['res']
    inputs = res['inputs']
    st.divider()
    
    # Red Flags
    flags = check_red_flags(inputs)
    if flags:
        with st.expander(f"🚩 ALERTAS DE AUDITORÍA ({len(flags)})", expanded=True):
            for f in flags: st.markdown(f"<div class='red-flag'>{f}</div>", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["📊 Dashboard", "⚖️ Comparables", "🔮 Escenarios", "🎲 Montecarlo", "📄 Reporte"])
    
    with t1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Enterprise Value", f"${res['ev']/1e6:,.1f} M")
        c2.metric("Deuda Neta", f"${res['debt']/1e6:,.1f} M")
        c3.metric("Equity Value", f"${res['equity']/1e6:,.1f} M", delta="DCF")
        
        fig = go.Figure(go.Waterfall(
            orientation="v", measure=["relative", "relative", "total", "relative", "total"],
            x=["VP Flujos", "VP Terminal", "EV", "(-) Deuda", "Equity"],
            y=[res['vp_flows'], res['vp_tv'], 0, -res['debt'], 0],
            text=[f"{x/1e6:.1f}" for x in [res['vp_flows'], res['vp_tv'], res['ev'], -res['debt'], res['equity']]],
            connector={"line":{"color":"#333"}}
        ))
        fig.update_layout(height=300, title="Puente de Valoración (M$)")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        val_ev = inputs['peer_ev_ebitda'] * res['ebitda_t1'] - inputs['debt_t0']
        val_pe = inputs['peer_pe'] * res['net_income_t1']
        
        c1, c2 = st.columns(2)
        c1.metric("Según EV/EBITDA", f"${val_ev/1e6:,.1f} M", delta=f"{(val_ev/res['equity']-1):.1%}")
        c2.metric("Según P/E", f"${val_pe/1e6:,.1f} M", delta=f"{(val_pe/res['equity']-1):.1%}")
        
        if audit_mode:
            st.info(f"EBITDA Proy: ${res['ebitda_t1']/1e6:.1f}M | Utilidad Neta Proy: ${res['net_income_t1']/1e6:.1f}M")

    with t3:
        if 'scenarios' in st.session_state:
            df_s = st.session_state['scenarios'].copy()
            df_s['Equity ($M)'] = df_s['Equity Value']/1e6
            st.dataframe(df_s[['Escenario', 'Equity ($M)']].style.format({"Equity ($M)": "${:,.1f}"}), use_container_width=True, hide_index=True)

    with t4:
        st.subheader("Análisis de Riesgo Estocástico")
        if st.button("🎲 Correr Simulación"):
            sim_wacc = np.maximum(np.random.normal(res['wacc'], 0.015, 5000), 0.01)
            sim_g = np.random.normal(res['g_term'], 0.005, 5000)
            sim_wacc = np.maximum(sim_wacc, sim_g + 0.005)
            tv_s = res['fcff_last']*(1+sim_g)/(sim_wacc-sim_g)
            ev_s = (tv_s / ((1+sim_wacc)**5)) + res['vp_flows']
            st.session_state['sim_data'] = ev_s - res['debt']

        if 'sim_data' in st.session_state:
            data = st.session_state['sim_data']
            mean_val = np.mean(data)
            var_5 = np.percentile(data, 5)
            prob_loss = np.sum(data < 0) / len(data) * 100
            
            fig = px.histogram(x=data/1e6, nbins=40, title="Distribución de Probabilidad")
            fig.add_vline(x=mean_val/1e6, line_color="green")
            fig.add_vline(x=var_5/1e6, line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            
            # MEJORA: Explicación y Conclusiones
            st.markdown(f"""
            ### 📝 Dictamen de Riesgos:
            1.  **Valor Central (Esperado):** **${mean_val/1e6:,.1f} M**.
                *Es el valor más probable promediando 5,000 futuros posibles.*
            2.  **Escenario Adverso (VaR 5%):** **${var_5/1e6:,.1f} M**.
                *Existe un 5% de probabilidad de que el valor real sea menor a esta cifra. Útil para medir el "peor caso razonable".*
            3.  **Probabilidad de Insolvencia:** **{prob_loss:.1f}%**.
                *Probabilidad de que el Equity sea negativo (Deuda > Valor Empresa).*
            """)

    with t5:
        st.subheader("Generación de Entregables")
        if st.button("📥 Descargar Reporte Completo (.xlsx)"):
            out = io.BytesIO()
            wb = xlsxwriter.Workbook(out, {'in_memory': True, 'nan_inf_to_errors': True})
            
            # Formatos
            fmt_head = wb.add_format({'bold': True, 'bg_color': '#002060', 'font_color': 'white'})
            fmt_curr = wb.add_format({'num_format': '$ #,##0'})
            
            # 1. RESUMEN
            ws = wb.add_worksheet("Resumen Gerencial")
            ws.write('A1', f"VALORACIÓN: {proyecto}", fmt_head)
            ws.write('A3', "MÉTRICAS CLAVE", fmt_head)
            ws.write('A4', "Enterprise Value"); ws.write('B4', res['ev'], fmt_curr)
            ws.write('A5', "Equity Value"); ws.write('B5', res['equity'], fmt_curr)
            
            if 'scenarios' in st.session_state:
                ws.write('A8', "ESCENARIOS", fmt_head)
                df_scen = st.session_state['scenarios']
                for r, row in enumerate(df_scen.values):
                    ws.write(r+9, 0, row[0])
                    ws.write(r+9, 1, row[1], fmt_curr)
            
            if flags:
                ws.write('D3', "ALERTAS (RED FLAGS)", fmt_head)
                for i, f in enumerate(flags): ws.write(i+4, 3, f)

            # 2. MODELO DCF
            ws_dcf = wb.add_worksheet("Modelo DCF")
            ws_dcf.write('A1', "Flujo de Caja Descontado", fmt_head)
            df = res['df'].fillna(0)
            for c, col in enumerate(df.columns): ws_dcf.write(2, c, col, fmt_head)
            for r, row in enumerate(df.values):
                for c, val in enumerate(row): ws_dcf.write(r+3, c, val, fmt_curr if c>0 else None)

            # 3. MÚLTIPLOS
            ws_rel = wb.add_worksheet("Valor Relativo")
            ws_rel.write('A1', "Valoración por Múltiplos", fmt_head)
            ws_rel.write('A3', "Método"); ws_rel.write('B3', "Valor Equity", fmt_head)
            ws_rel.write('A4', "EV/EBITDA"); ws_rel.write('B4', val_ev, fmt_curr)
            ws_rel.write('A5', "P/E Ratio"); ws_rel.write('B5', val_pe, fmt_curr)

            # 4. RIESGO (Con Gráfico)
            if 'sim_data' in st.session_state:
                ws_risk = wb.add_worksheet("Riesgo Montecarlo")
                data_sim = st.session_state['sim_data']
                counts, bin_edges = np.histogram(data_sim, bins=30)
                
                ws_risk.write('A1', "Distribución de Frecuencias", fmt_head)
                for i in range(len(counts)):
                    ws_risk.write(i+2, 0, f"{bin_edges[i]/1e6:.0f}M - {bin_edges[i+1]/1e6:.0f}M")
                    ws_risk.write(i+2, 1, counts[i])
                
                chart = wb.add_chart({'type': 'column'})
                chart.add_series({'categories': ['Riesgo Montecarlo', 2, 0, len(counts)+1, 0], 'values': ['Riesgo Montecarlo', 2, 1, len(counts)+1, 1]})
                chart.set_title({'name': 'Distribución de Valor Equity'})
                ws_risk.insert_chart('E2', chart)

            wb.close()
            out.seek(0)
            st.download_button("Bajar Excel Completo", data=out, file_name=f"Reporte_{proyecto}.xlsx")

st.markdown("<div class='footer'>Valuation Master Suite v4.1 © 2026 - USIL</div>", unsafe_allow_html=True)
