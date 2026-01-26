import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import xlsxwriter
from datetime import datetime

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Valuation Master Suite - USIL",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS (Minimalista para evitar conflictos) ---
st.markdown("""
    <style>
    .footer {
        position: fixed; bottom: 0; left: 0; width: 100%; 
        background-color: transparent; color: #888; 
        text-align: center; padding: 10px; font-size: 12px; 
        z-index: 999;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LÓGICA FINANCIERA (BACKEND) ---
def calculate_dcf(inputs):
    # Desempaquetar variables
    tax = inputs['tax_rate'] / 100
    rf, erp, kd = inputs['rf']/100, inputs['erp']/100, inputs['kd']/100
    g_sales, g_term = inputs['g_sales']/100, inputs['g_term']/100
    capex_pct = inputs['capex']/100
    
    # --- CÁLCULO WACC (Desglose) ---
    beta_l = inputs['beta_u'] * (1 + (1-tax)*inputs['de_target'])
    ke = rf + (beta_l * erp)
    kd_net = kd * (1 - tax)
    wd = inputs['de_target'] / (1 + inputs['de_target'])
    we = 1 / (1 + inputs['de_target'])
    wacc = (we * ke) + (wd * kd_net)

    # Validación de Auditoría
    if wacc <= g_term:
        return {'error': f"⚠️ Error Crítico de Supuestos: El WACC ({wacc:.1%}) es menor o igual al crecimiento perpetuo (g={g_term:.1%}). Esto matemático imposibilita la valoración."}

    # --- PROYECCIONES ---
    schedule = []
    nwc_prev = inputs['ar_t0'] + inputs['inv_t0'] - inputs['ap_t0']
    curr_sales = inputs['sales_t0']
    curr_debt = inputs['debt_t0']
    
    # Ratios base (Históricos)
    margin_base = (inputs['sales_t0'] - inputs['cogs_t0'] - inputs['opex_t0']) / inputs['sales_t0']
    dso = (inputs['ar_t0'] / inputs['sales_t0']) * 360 if inputs['sales_t0'] > 0 else 0
    dio = (inputs['inv_t0'] / inputs['cogs_t0']) * 360 if inputs['cogs_t0'] > 0 else 0
    dpo = (inputs['ap_t0'] / inputs['cogs_t0']) * 360 if inputs['cogs_t0'] > 0 else 0
    
    schedule.append({"Año": 0, "Ventas": inputs['sales_t0'], "EBITDA": inputs['sales_t0']*margin_base, "FCFF": 0, "FCFE": 0})
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
        
        interest = curr_debt * kd * (1-tax)
        net_borrow = inputs['debt_new'] - inputs['debt_amort']
        fcfe = fcff - interest + net_borrow
        curr_debt = curr_debt - inputs['debt_amort'] + inputs['debt_new']
        
        schedule.append({
            "Año": i, "Ventas": curr_sales, "EBITDA": ebitda,
            "NOPAT": nopat, "Capex": capex, "Var NWC": var_nwc,
            "FCFF": fcff, "FCFE": fcfe
        })

    # --- VALORACIÓN ---
    tv = fcff_list[-1] * (1+g_term) / (wacc - g_term)
    vp_flows = sum([f / ((1+wacc)**(i+1)) for i, f in enumerate(fcff_list)])
    vp_tv = tv / ((1+wacc)**5)
    ev = vp_flows + vp_tv
    equity_val = ev - inputs['debt_t0']

    return {
        'wacc': wacc, 'ke': ke, 'kd_net': kd_net, 'beta_l': beta_l, # Agregados para reporte
        'ev': ev, 'equity': equity_val, 'debt': inputs['debt_t0'],
        'vp_flows': vp_flows, 'vp_tv': vp_tv, 
        'df': pd.DataFrame(schedule),
        'g_term': g_term, 'fcff_last': fcff_list[-1], 'g_sales': g_sales,
        'inputs': inputs 
    }

# --- 3. PANEL DE CONTROL (SIDEBAR) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Logo_USIL.png/640px-Logo_USIL.png", width=140)
    st.title("Panel de Control")
    st.markdown("**Prof. Jorge Rojas (MBA)**")
    
    alumno = st.text_input("Analista / Alumno", value="Estudiante MBA")
    proyecto = st.text_input("Empresa Target", value="Empresa Modelo S.A.")
    
    with st.expander("⚙️ Tasas de Mercado (CAPM)", expanded=True):
        rf_in = st.number_input("Tasa Libre Riesgo (Rf %)", value=4.0, help="Rendimiento de Bonos del Tesoro")
        erp_in = st.number_input("Prima Riesgo (ERP %)", value=5.5, help="Equity Risk Premium")
        beta_u = st.number_input("Beta Desapalancado", value=0.90, help="Riesgo sistemático de la industria")
        kd_in = st.number_input("Costo Deuda (Kd %)", value=7.5, help="Tasa de interés bancaria")

# --- 4. INTERFAZ DE USUARIO ---
st.title(f"💎 Valuation Suite: {proyecto}")

tab_in1, tab_in2 = st.tabs(["1️⃣ Estados Financieros", "2️⃣ Supuestos de Negocio"])
inputs = {}

with tab_in1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("P&L (Estado de Resultados)")
        inputs['sales_t0'] = st.number_input("Ventas ($)", value=10000000.0)
        inputs['cogs_t0'] = st.number_input("Costo Ventas ($)", value=6000000.0)
        inputs['opex_t0'] = st.number_input("Gastos Op. ($)", value=1500000.0)
        inputs['deprec_t0'] = st.number_input("Depreciación ($)", value=400000.0)
        inputs['tax_rate'] = st.slider("Tax Corporativo (%)", 0, 40, 25)
    with c2:
        st.subheader("Balance General")
        inputs['ar_t0'] = st.number_input("Ctas Cobrar", value=830000.0)
        inputs['inv_t0'] = st.number_input("Inventarios", value=750000.0)
        inputs['ap_t0'] = st.number_input("Ctas Pagar", value=600000.0)
        inputs['debt_t0'] = st.number_input("Deuda Financiera Total", value=2000000.0)
        inputs['equity_book_t0'] = st.number_input("Patrimonio Contable", value=3000000.0)

with tab_in2:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Crecimiento**")
        inputs['g_sales'] = st.number_input("Crec. Ventas (%)", value=5.0)
        inputs['g_term'] = st.number_input("Crec. Perpetuo (g) (%)", value=2.5)
        inputs['capex'] = st.number_input("Capex (% Ventas)", value=4.5)
    with c2:
        st.markdown("**Estructura Capital**")
        inputs['de_target'] = st.number_input("Target D/E", value=0.40)
    with c3:
        st.markdown("**Deuda**")
        inputs['debt_amort'] = st.number_input("Amort. Deuda ($)", value=200000.0)
        inputs['debt_new'] = st.number_input("Nueva Deuda ($)", value=50000.0)

inputs.update({'rf': rf_in, 'erp': erp_in, 'beta_u': beta_u, 'kd': kd_in})

if st.button("🚀 EJECUTAR MODELO DE VALORACIÓN", type="primary", use_container_width=True):
    results = calculate_dcf(inputs)
    if 'error' in results:
        st.error(results['error'])
    else:
        st.session_state['res'] = results
        # Limpiar simulación previa para evitar inconsistencias
        if 'sim_data' in st.session_state: del st.session_state['sim_data']

# --- 5. RESULTADOS Y ANÁLISIS ---
if 'res' in st.session_state:
    res = st.session_state['res']
    st.markdown("---")
    
    t1, t2, t3, t4 = st.tabs(["📊 Dashboard Ejecutivo", "🌪️ Análisis de Sensibilidad", "🎲 Riesgo Montecarlo", "📄 Reporte Excel"])
    
    # --- TAB 1: DASHBOARD ---
    with t1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Enterprise Value", f"${res['ev']/1e6:,.1f} M")
        c2.metric("Deuda Neta", f"${res['debt']/1e6:,.1f} M")
        c3.metric("Equity Value", f"${res['equity']/1e6:,.1f} M", delta="Valor Final")
        c4.metric("WACC", f"{res['wacc']:.2%}", help="Costo Promedio Ponderado de Capital")
        
        st.subheader("Puente de Valoración (Waterfall)")
        fig = go.Figure(go.Waterfall(
            orientation = "v",
            measure = ["relative", "relative", "total", "relative", "total"],
            x = ["VP Flujos Operativos", "VP Valor Terminal", "Enterprise Value", "(-) Deuda Financiera", "Equity Value"],
            text = [f"{res['vp_flows']/1e6:.1f}", f"{res['vp_tv']/1e6:.1f}", f"{res['ev']/1e6:.1f}", f"-{res['debt']/1e6:.1f}", f"{res['equity']/1e6:.1f}"],
            y = [res['vp_flows'], res['vp_tv'], 0, -res['debt'], 0],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig.update_layout(template="streamlit", height=350, title="Desglose de Valor (Millones USD)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Detalle de Flujos de Caja Proyectados**")
        st.dataframe(res['df'].style.format("{:,.0f}"), use_container_width=True, hide_index=True)

    # --- TAB 2: SENSIBILIDAD (MEJORADO: EJES CLAROS) ---
    with t2:
        st.subheader("Mapa de Sensibilidad: WACC vs Crecimiento (g)")
        st.caption("Este gráfico muestra cómo varía el valor del Equity ante cambios en las variables críticas.")
        
        w_range = np.linspace(res['wacc']*0.8, res['wacc']*1.2, 5)
        g_range = np.linspace(res['g_term']*0.5, res['g_term']*1.5, 5)
        
        # Matriz de valores
        z = [[( (res['fcff_last']*(1+g)/(w-g))/((1+w)**5) + res['vp_flows']*(res['wacc']/w) - res['debt'] )/1e6 if w>g else 0 for w in w_range] for g in g_range]
        
        fig_h = go.Figure(data=go.Heatmap(
            z=z, 
            x=[f"{x:.1%}" for x in w_range], 
            y=[f"{y:.1%}" for y in g_range], 
            colorscale='RdYlGn',
            hovertemplate = "WACC: %{x}<br>g: %{y}<br>Equity: $%{z:.1f}M<extra></extra>"
        ))
        
        # CORRECCIÓN SOLICITADA: EJES EXPLICADOS
        fig_h.update_layout(
            title="Sensibilidad del Equity Value (Millones $)",
            xaxis_title="Costo de Capital (WACC %)",
            yaxis_title="Crecimiento a Perpetuidad (g %)",
            height=500
        )
        st.plotly_chart(fig_h, use_container_width=True)

    # --- TAB 3: MONTECARLO ---
    with t3:
        st.subheader("Simulación de Riesgo (Montecarlo)")
        col_m1, col_m2 = st.columns([1, 3])
        
        with col_m1:
            st.markdown("Genera miles de escenarios aleatorios variando el WACC y el Crecimiento para medir la probabilidad de éxito.")
            if st.button("🎲 Correr 5,000 Escenarios"):
                sim_wacc = np.random.normal(res['wacc'], 0.015, 5000)
                sim_g = np.random.normal(res['g_term'], 0.005, 5000)
                sim_wacc = np.maximum(sim_wacc, sim_g + 0.005) # Evitar división por cero
                tv_s = res['fcff_last']*(1+sim_g)/(sim_wacc-sim_g)
                ev_s = (tv_s / ((1+sim_wacc)**5)) + res['vp_flows']
                st.session_state['sim_data'] = ev_s - res['debt']

        with col_m2:
            if 'sim_data' in st.session_state:
                data = st.session_state['sim_data']
                mean_val = np.mean(data)
                var_5 = np.percentile(data, 5)
                prob_loss = np.sum(data < 0) / len(data) * 100
                
                fig_hist = px.histogram(x=data/1e6, nbins=40, title="Distribución Probabilística del Valor")
                fig_hist.add_vline(x=mean_val/1e6, line_color="green", annotation_text="Media Esperada")
                fig_hist.add_vline(x=var_5/1e6, line_color="red", annotation_text="VaR 5% (Riesgo)")
                fig_hist.update_layout(xaxis_title="Equity Value (Millones $)", yaxis_title="Frecuencia", template="streamlit")
                st.plotly_chart(fig_hist, use_container_width=True)
                
                st.info(f"💡 Interpretación: Existe una probabilidad del **{prob_loss:.1f}%** de que el valor del patrimonio sea negativo (insolvencia técnica). En el peor 5% de los casos, el valor cae por debajo de **${var_5/1e6:,.1f} M**.")
            else:
                st.warning("👈 Presiona el botón para ejecutar la simulación.")

    # --- TAB 4: EXCEL MEJORADO (DETALLE WACC) ---
    with t4:
        st.subheader("Reporte Auditoría (Excel)")
        st.markdown("Descarga el informe completo, incluyendo el desglose de tasas y gráficos.")
        
        if st.button("📥 Descargar Reporte Completo (.xlsx)"):
            output = io.BytesIO()
            workbook = xlsxwriter.Workbook(output, {'in_memory': True, 'nan_inf_to_errors': True})
            
            # --- ESTILOS EXCEL ---
            fmt_head_main = workbook.add_format({'bold': True, 'bg_color': '#002060', 'font_color': 'white', 'font_size': 12, 'border': 1})
            fmt_subhead = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
            fmt_curr = workbook.add_format({'num_format': '$ #,##0', 'border': 1})
            fmt_pct = workbook.add_format({'num_format': '0.00%', 'border': 1, 'align': 'center'})
            fmt_decimal = workbook.add_format({'num_format': '0.00', 'border': 1, 'align': 'center'})
            fmt_border = workbook.add_format({'border': 1})
            
            # === HOJA 1: RESUMEN EJECUTIVO ===
            ws = workbook.add_worksheet("Resumen")
            ws.set_column('A:A', 30)
            ws.set_column('B:B', 15)
            
            ws.merge_range('A1:B1', f"VALORACIÓN: {proyecto.upper()}", fmt_head_main)
            ws.write('A2', "Fecha:", fmt_border)
            ws.write('B2', datetime.now().strftime("%Y-%m-%d"), fmt_border)
            ws.write('A3', "Analista:", fmt_border)
            ws.write('B3', alumno, fmt_border)
            
            # Sección WACC Detallada (CORRECCIÓN SOLICITADA)
            ws.write('A5', "DETALLE TASA DE DESCUENTO (WACC)", fmt_head_main)
            wacc_data = [
                ("Tasa Libre de Riesgo (Rf)", res['inputs']['rf']/100, fmt_pct),
                ("Beta Desapalancado (Unlevered)", res['inputs']['beta_u'], fmt_decimal),
                ("Ratio D/E Objetivo", res['inputs']['de_target'], fmt_decimal),
                ("Tax Rate", res['inputs']['tax_rate']/100, fmt_pct),
                ("Beta Reapalancado (Levered)", res['beta_l'], fmt_decimal),
                ("Equity Risk Premium (ERP)", res['inputs']['erp']/100, fmt_pct),
                ("Costo del Equity (Ke)", res['ke'], fmt_pct),
                ("Costo de Deuda Bruto (Kd)", res['inputs']['kd']/100, fmt_pct),
                ("Costo de Deuda Neto (Kd * (1-t))", res['kd_net'], fmt_pct),
                ("WACC FINAL", res['wacc'], fmt_pct)
            ]
            
            row = 6
            for label, val, fmt in wacc_data:
                ws.write(row, 0, label, fmt_subhead if "WACC" in label else fmt_border)
                ws.write(row, 1, val, fmt)
                row += 1
                
            # Sección Valoración
            ws.write(row + 2, 0, "RESULTADOS DE VALORACIÓN", fmt_head_main)
            val_data = [
                ("VP Flujos Operativos", res['vp_flows'], fmt_curr),
                ("VP Valor Terminal", res['vp_tv'], fmt_curr),
                ("ENTERPRISE VALUE (EV)", res['ev'], fmt_curr),
                ("(-) Deuda Financiera", res['debt'], fmt_curr),
                ("EQUITY VALUE", res['equity'], fmt_curr)
            ]
            
            row += 3
            for label, val, fmt in val_data:
                ws.write(row, 0, label, fmt_subhead if "EQUITY" in label else fmt_border)
                ws.write(row, 1, val, fmt)
                row += 1

            # === HOJA 2: FLUJOS ===
            ws_cf = workbook.add_worksheet("Flujos Proyectados")
            ws_cf.set_column('A:Z', 15)
            ws_cf.write(0, 0, "Proyección Financiera (5 Años)", fmt_head_main)
            
            df = res['df'].fillna(0)
            # Headers
            for c, col_name in enumerate(df.columns):
                ws_cf.write(1, c, col_name, fmt_subhead)
            # Data
            for r, row_data in enumerate(df.values):
                for c, val in enumerate(row_data):
                    ws_cf.write(r+2, c, val, fmt_curr if c > 0 else fmt_border)
            
            # === HOJA 3: GRÁFICO MONTECARLO ===
            if 'sim_data' in st.session_state:
                ws_sim = workbook.add_worksheet("Simulación Riesgo")
                data_sim = st.session_state['sim_data']
                counts, bin_edges = np.histogram(data_sim, bins=30)
                
                ws_sim.write('A1', "Distribución de Frecuencias (Montecarlo)", fmt_head_main)
                ws_sim.write('A2', "Rango ($)", fmt_subhead)
                ws_sim.write('B2', "Frecuencia", fmt_subhead)
                
                for i in range(len(counts)):
                    ws_sim.write(i+2, 0, f"{bin_edges[i]/1e6:.1f}M - {bin_edges[i+1]/1e6:.1f}M")
                    ws_sim.write(i+2, 1, counts[i])
                
                chart = workbook.add_chart({'type': 'column'})
                chart.add_series({
                    'name': 'Frecuencia',
                    'categories': ['Simulación Riesgo', 2, 0, len(counts)+1, 0],
                    'values':     ['Simulación Riesgo', 2, 1, len(counts)+1, 1],
                    'gap':        5,
                    'fill':       {'color': '#4472C4'}
                })
                chart.set_title({'name': 'Distribución de Equity Value'})
                chart.set_x_axis({'name': 'Rango de Valor (Millones)'})
                ws_sim.insert_chart('E2', chart)

            workbook.close()
            output.seek(0)
            st.download_button("Bajar .xlsx", data=output, file_name=f"Valuation_{proyecto}_V10.xlsx")

# Footer
st.markdown("<div class='footer'>Valuation Master Suite © 2026 - Desarrollado para USIL por Prof. Jorge Rojas</div>", unsafe_allow_html=True)
