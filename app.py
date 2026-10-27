import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve

# ==========================================
# 1. CONFIGURACIÓN Y ESTILOS AVANZADOS
# ==========================================
st.set_page_config(layout="wide", page_title="AgroTermo Pro", page_icon="🌱")

# Inyección de CSS para diseño profesional (Tarjetas, Pestañas y Títulos)
st.markdown("""
<style>
    /* Estilo de las tarjetas de métricas */
    div[data-testid="metric-container"] {
        background-color: rgba(46, 125, 50, 0.1);
        border: 1px solid rgba(46, 125, 50, 0.5);
        padding: 15px 20px;
        border-radius: 15px;
        border-left: 6px solid #2e7d32;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    /* Color del valor numérico */
    div[data-testid="stMetricValue"] > div { 
        color: #2e7d32; 
        font-weight: 800;
    }
    /* Estilizar las Pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(0, 0, 0, 0.03);
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(46, 125, 50, 0.1);
        border-bottom: 4px solid #2e7d32 !important;
        color: #2e7d32;
    }
    /* Título principal centrado y estilizado */
    .main-title {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        color: #1b5e20;
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BASE DE DATOS Y FUNCIONES FÍSICAS
# ==========================================
GASES_PROPS = {
    "Dióxido de Carbono (CO2)": (304.2, 73.8, 0.224, 0.04401),
    "Agua (H2O)":               (647.1, 220.6, 0.344, 0.018015),
    "Nitrógeno (N2)":           (126.2, 34.0, 0.037, 0.02801),
    "Etanol (C2H5OH)":          (513.9, 61.4, 0.644, 0.04607),
    "Metano (CH4)":             (190.6, 46.1, 0.011, 0.01604),
    "Amoníaco (NH3)":           (405.4, 113.5, 0.253, 0.01703), 
    "Propano (C3H8)":           (369.8, 42.5, 0.152, 0.04410)
}

R_bar = 0.08314 
R_J = 8.314     

def parametros_eos(Tc, Pc, w, modelo="vdw"):
    if modelo == "vdw":
        return (27 * (R_bar * Tc)**2) / (64 * Pc), (R_bar * Tc) / (8 * Pc)
    else:
        a = (0.45724 * (R_bar * Tc)**2) / Pc
        b = (0.07780 * R_bar * Tc) / Pc
        kappa = 0.37464 + 1.54226 * w - 0.26992 * w**2
        return a, b, kappa

def maxwell_vdw(T, a, b):
    def sistema(vars):
        Vl, Vv, Psat = vars
        P_vl = (R_bar * T) / (Vl - b) - a / (Vl**2)
        P_vv = (R_bar * T) / (Vv - b) - a / (Vv**2)
        int_vdw = R_bar * T * np.log((Vv - b) / (Vl - b)) + a * (1/Vv - 1/Vl)
        return [P_vl - Psat, P_vv - Psat, int_vdw - Psat * (Vv - Vl)]
    
    try:
        sol = fsolve(sistema, [b * 1.5, (R_bar * T) / (0.5 * a / b**2) + b, 0.1])
        return sol[0], sol[1], sol[2]
    except:
        return np.nan, np.nan, np.nan

def calc_P(V, T, modelo_key, a, b, kappa=None, Tc=None):
    if modelo_key == "vdw":
        return (R_bar * T) / (V - b) - (a / V**2)
    else:
        alpha = (1 + kappa * (1 - np.sqrt(T / Tc)))**2
        return (R_bar * T) / (V - b) - (a * alpha) / (V**2 + 2*b*V - b**2)

# ==========================================
# 3. INTERFAZ DE USUARIO (UI)
# ==========================================
st.markdown("<h1 class='main-title'>🌱 AgroTermo Pro: Simulador Fisicoquímico</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; margin-bottom: 30px;'>Plataforma interactiva para el análisis termodinámico y diseño de procesos agroindustriales.</p>", unsafe_allow_html=True)

# Panel Lateral
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2153/2153106.png", width=100) # Ícono decorativo genérico
st.sidebar.header("⚙️ Configuración Global")

gas = st.sidebar.selectbox("Selecciona el Fluido de Trabajo:", list(GASES_PROPS.keys()))
modelo = st.sidebar.selectbox("Ecuación de Estado:", ["Van der Waals", "Peng-Robinson"])

st.sidebar.markdown("---")
st.sidebar.subheader("Aplicación Industrial")

TIPS_AGRO = {
    "Dióxido de Carbono (CO2)": "El CO2 supercrítico es vital para extraer aceites esenciales o descafeinar café (baja degradación térmica).",
    "Agua (H2O)": "Vapor sobrecalentado: principal medio de transferencia térmica en pasteurización y marmitas.",
    "Nitrógeno (N2)": "Líquido criogénico para congelamiento ultra rápido (IQF). Gas para envasado en atmósfera modificada.",
    "Etanol (C2H5OH)": "Solvente GRAS para extracción de pigmentos y fluido secundario en refrigeración.",
    "Metano (CH4)": "Base del biogás agroindustrial para cogeneración de energía térmica.",
    "Amoníaco (NH3)": "El refrigerante rey (R-717) de la cadena de frío en frigoríficos de agroexportación.",
    "Propano (C3H8)": "Refrigerante ecológico (R-290) que reemplaza a los freones por su bajo impacto global."
}

st.sidebar.info(TIPS_AGRO[gas])
st.sidebar.markdown("---")
st.sidebar.caption("👨‍💻 Desarrollado para Ingeniería Agroindustrial")

# Variables Termodinámicas del Gas
Tc, Pc, w, MasaMolar = GASES_PROPS[gas]
modelo_key = "vdw" if modelo == "Van der Waals" else "pr"
Vc = (3 * R_bar * Tc) / (8 * Pc) if modelo_key == "vdw" else 0.307 * R_bar * Tc / Pc

if modelo_key == "vdw":
    a, b = parametros_eos(Tc, Pc, w, "vdw")
    kappa = None
else:
    a, b, kappa = parametros_eos(Tc, Pc, w, "pr")

# Tarjetas superiores globales
col1, col2, col3 = st.columns(3)
col1.metric("Temperatura Crítica (Tc)", f"{Tc} K")
col2.metric("Presión Crítica (Pc)", f"{Pc} bar")
col3.metric("Masa Molar (M)", f"{MasaMolar*1000:.2f} g/mol")
st.markdown("<br>", unsafe_allow_html=True)

# Navegación por pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Gases Reales (2D)", 
    "🧊 Superficie P-V-T (3D)", 
    "🚀 Cinética (Maxwell)", 
    "❄️ Diseño de Frío",
    "🥔 Termoquímica Poscosecha"
])

# --- PESTAÑA 1: Gráficos 2D ---
with tab1:
    st.subheader("Análisis de Isotermas y Factor de Compresibilidad")
    colA, colB = st.columns(2)
    V_arr = np.linspace(1.05 * b, 8 * Vc, 500)
    T_sub, T_crit, T_sup = Tc * 0.85, Tc, Tc * 1.15
    
    P_sub = calc_P(V_arr, T_sub, modelo_key, a, b, kappa, Tc)
    P_crit = calc_P(V_arr, T_crit, modelo_key, a, b, kappa, Tc)
    P_sup = calc_P(V_arr, T_sup, modelo_key, a, b, kappa, Tc)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=V_arr, y=P_sub, name=f'Subcrítica ({T_sub:.1f} K)', line=dict(color='#4caf50', width=3)))
    
    if modelo_key == "vdw":
        Vl, Vv, Psat = maxwell_vdw(T_sub, a, b)
        if not np.isnan(Psat):
            fig1.add_trace(go.Scatter(x=[Vl, Vv], y=[Psat, Psat], name='Condensación (Maxwell)', line=dict(color='gray', width=2, dash='dash')))
        else:
            st.warning(f"La isoterma de Maxwell para {T_sub:.1f} K no convergió matemáticamente.")
            
    fig1.add_trace(go.Scatter(x=V_arr, y=P_crit, name=f'Crítica ({T_crit:.1f} K)', line=dict(color='#ff9800', width=3)))
    fig1.add_trace(go.Scatter(x=V_arr, y=P_sup, name=f'Supercrítica ({T_sup:.1f} K)', line=dict(color='#e91e63', width=3)))
    fig1.add_trace(go.Scatter(x=[Vc], y=[Pc], mode='markers', name='Punto Crítico', marker=dict(color='red', size=10, symbol='star')))
    
    fig1.update_layout(title="Diagrama de Fases: Presión vs Volumen", xaxis_title="Volumen (L/mol)", yaxis_title="Presión (bar)", 
                       yaxis=dict(range=[0, Pc * 2]), xaxis=dict(range=[0, Vc * 5]), hovermode="x unified",
                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    with colA: 
        st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=P_sub, y=(P_sub * V_arr) / (R_bar * T_sub), name=f'T = {T_sub:.1f} K', line=dict(color='#4caf50', width=3)))
    fig2.add_trace(go.Scatter(x=P_crit, y=(P_crit * V_arr) / (R_bar * T_crit), name=f'T = {T_crit:.1f} K', line=dict(color='#ff9800', width=3)))
    fig2.add_shape(type="line", x0=0, y0=1, x1=Pc*2, y1=1, line=dict(color="gray", width=2, dash="dash"))
    
    fig2.update_layout(title="Desviación de la Idealidad (Z)", xaxis_title="Presión (bar)", yaxis_title="Factor de Compresibilidad (Z)", 
                       yaxis=dict(range=[0, 1.2]), xaxis=dict(range=[0, Pc * 2]), hovermode="x unified",
                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    with colB: 
        st.plotly_chart(fig2, use_container_width=True)

# --- PESTAÑA 2: Superficie 3D ---
with tab2:
    st.subheader("Campana Termodinámica en 3 Dimensiones")
    st.markdown("Explora el comportamiento del fluido en el espacio interactivo P-V-T. Usa el ratón para rotar y hacer zoom.")
    
    V_grid, T_grid = np.meshgrid(np.linspace(1.2 * b, 5 * Vc, 50), np.linspace(Tc * 0.7, Tc * 1.3, 50))
    P_mesh = np.clip(calc_P(V_grid, T_grid, modelo_key, a, b, kappa, Tc), 0, Pc * 3)
    
    fig3 = go.Figure(data=[go.Surface(z=P_mesh, x=V_grid, y=T_grid, colorscale='Viridis', opacity=0.9)])
    fig3.add_trace(go.Scatter3d(x=[Vc], y=[Tc], z=[Pc], mode='markers', name='Punto Crítico', marker=dict(color='red', size=8)))
    
    fig3.update_layout(scene=dict(xaxis_title='Volumen', yaxis_title='Temperatura', zaxis_title='Presión'), 
                       margin=dict(l=0, r=0, b=0, t=0), height=600)
    st.plotly_chart(fig3, use_container_width=True)

# --- PESTAÑA 3: Maxwell-Boltzmann ---
with tab3:
    st.subheader(f"Cinética Molecular del {gas.split()[0]}")
    st.write("Simulación de la distribución de velocidades de Maxwell-Boltzmann en función de la energía térmica.")
    
    temp_mb = st.slider("🌡️ Modifica la Temperatura del Gas (Kelvin):", min_value=100, max_value=1500, value=int(Tc), step=50)
    
    vp = np.sqrt((2 * R_J * temp_mb) / MasaMolar)          
    v_prom = np.sqrt((8 * R_J * temp_mb) / (np.pi * MasaMolar)) 
    v_rms = np.sqrt((3 * R_J * temp_mb) / MasaMolar)       
    
    v = np.linspace(0, v_rms * 3, 1000)
    fv = 4 * np.pi * (MasaMolar / (2 * np.pi * R_J * temp_mb))**(1.5) * (v**2) * np.exp(-MasaMolar * (v**2) / (2 * R_J * temp_mb))
    
    fig_mb = go.Figure()
    fig_mb.add_trace(go.Scatter(x=v, y=fv, fill='tozeroy', mode='lines', line=dict(color='#03a9f4', width=3), name="Distribución"))
    
    fig_mb.add_vline(x=vp, line_dash="dash", line_color="#ff9800", annotation_text=f"Más probable ({vp:.0f} m/s)", annotation_position="top left")
    fig_mb.add_vline(x=v_rms, line_dash="dash", line_color="#e91e63", annotation_text=f"RMS ({v_rms:.0f} m/s)", annotation_position="top right")
    
    fig_mb.update_layout(xaxis_title="Velocidad (m/s)", yaxis_title="Probabilidad", hovermode="x unified", showlegend=False,
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500)
    
    st.plotly_chart(fig_mb, use_container_width=True)

# --- PESTAÑA 4: Simulador de Enfriamiento Agroindustrial ---
with tab4:
    st.subheader("❄️ Proyección de Equilibrio Térmico (Ley Cero)")
    st.markdown("Diseña perfiles de enfriamiento para lotes de productos agroindustriales procesados.")
    
    col_input, col_grafico = st.columns([1, 2])
    
    with col_input:
        producto = st.selectbox("Selecciona la Matriz Alimentaria:", ["Néctar de Maracuyá", "Leche Cruda", "Pulpa de Mango"])
        
        if producto == "Néctar de Maracuyá":
            densidad, cp, t_ini_default, t_camara_default, k_enfriamiento = 1.05, 3.8, 25.0, 5.0, 0.02
        elif producto == "Leche Cruda":
            densidad, cp, t_ini_default, t_camara_default, k_enfriamiento = 1.03, 3.93, 30.0, 4.0, 0.015
        else:
            densidad, cp, t_ini_default, t_camara_default, k_enfriamiento = 1.01, 3.6, 20.0, -18.0, 0.025
            
        volumen = st.number_input("Volumen del lote (Litros):", min_value=10, max_value=5000, value=1000, step=100)
        t_inicial = st.slider("Temp. Inicial del Producto (°C):", -10.0, 90.0, t_ini_default)
        t_camara = st.slider("Temp. de la Cámara de Frío (°C):", -25.0, 30.0, t_camara_default)
        
        masa = volumen * densidad
        calor_remover = masa * cp * (t_inicial - t_camara)
        
    with col_grafico:
        tiempo_max = -np.log(0.01) / k_enfriamiento
        tiempo = np.linspace(0, tiempo_max, 200) 
        temperatura_t = t_camara + (t_inicial - t_camara) * np.exp(-k_enfriamiento * tiempo)
        
        fig_enfriamiento = go.Figure()
        fig_enfriamiento.add_trace(go.Scatter(x=tiempo, y=temperatura_t, mode='lines', 
                                              name=f'Perfil {producto}', line=dict(color='#2196f3', width=4)))
        
        fig_enfriamiento.add_hline(y=t_camara, line_dash="dash", line_color="#00bcd4", 
                                   annotation_text=f"Set Point Cámara ({t_camara}°C)", annotation_position="bottom right")
        
        fig_enfriamiento.update_layout(title="Curva de Aproximación al Equilibrio (Enfriamiento de Newton)",
                                       xaxis_title="Tiempo (Minutos)", yaxis_title="Temperatura (°C)",
                                       hovermode="x unified", height=400,
                                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_enfriamiento, use_container_width=True)

    st.markdown("---")
    res1, res2, res3 = st.columns(3)
    res1.success(f"**Masa del Lote:**\n\n### {masa:,.1f} kg")
    res2.warning(f"**Calor Sensible a Extraer (Q):**\n\n### {calor_remover:,.0f} kJ")
    res3.info(f"**Tiempo Estimado:**\n\n### {tiempo_max:.0f} min")

# --- PESTAÑA 5: Respiración Poscosecha (Ley de Hess) ---
with tab5:
    st.subheader("🥔 Carga Térmica por Respiración Poscosecha")
    st.markdown("Dimensionamiento de sistemas de refrigeración utilizando **Termoquímica (Ley de Hess)**.")
    
    with st.expander("🔬 Ver Sustento Matemático (Ley de Hess)"):
        st.markdown("Reacción de oxidación de la glucosa durante la respiración celular:")
        st.latex(r"C_6H_{12}O_6(s) + 6O_2(g) \rightarrow 6CO_2(g) + 6H_2O(l)")
        st.markdown("""
        **Entalpías Estándar de Formación:**
        * $\Delta H^\circ_f$ Glucosa = -1273.3 kJ/mol
        * $\Delta H^\circ_f CO_2(g)$ = -393.5 kJ/mol
        * $\Delta H^\circ_f H_2O(l)$ = -285.8 kJ/mol
        
        $\Delta H^\circ_{rxn} = [6(-393.5) + 6(-285.8)] - [-1273.3] = -2802.5 \text{ kJ/mol de glucosa}$
        
        Sabiendo que 1 mol de glucosa emite 6 moles de $CO_2$ (0.264 kg), la energía específica es:
        **$2802.5 \text{ kJ} / 0.264 \text{ kg} \approx 10,616 \text{ kJ por cada kg de } CO_2 \text{ emitido.}$**
        """)
    
    col_in, col_out = st.columns([1, 2])
    
    with col_in:
        TASA_PRODUCTOS = {
            "Papa (Almacenamiento largo)": 10.0,
            "Manzana (Climatérica - Baja)": 15.0,
            "Mango (Climatérica - Alta)": 40.0,
            "Espárrago (Perecedero - Alta)": 250.0,
            "Personalizado": 20.0
        }
        
        producto_sel = st.selectbox("Selecciona la Materia Prima Viva:", list(TASA_PRODUCTOS.keys()))
        masa_ton = st.slider("Tonelaje Almacenado (t):", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
        tasa_resp = st.number_input("Tasa de Respiración (mg CO2 / kg·h):", min_value=1.0, max_value=500.0, value=float(TASA_PRODUCTOS[producto_sel]))
        
        delta_H_CO2 = 10616  
        
    with col_out:
        masa_kg = masa_ton * 1000
        kg_CO2_hora = masa_kg * (tasa_resp / 1000000) 
        
        Q_kJ_h = kg_CO2_hora * delta_H_CO2
        Q_kW = Q_kJ_h / 3600  
        Q_TR = Q_kW / 3.517 # Toneladas de refrigeración
        Q_dia_MJ = (Q_kJ_h * 24) / 1000
        
        res_a, res_b, res_c = st.columns(3)
        res_a.metric("Potencia Térmica", f"{Q_kW:.2f} kW")
        res_b.metric("Emisión de CO2", f"{kg_CO2_hora:.2f} kg/h")
        res_c.metric("Equipo Requerido", f"{Q_TR:.2f} TR", help="1 TR (Tonelada de Refrigeración) = 3.517 kW")
        
        horas = np.linspace(0, 24, 100)
        calor_acumulado = (Q_kJ_h * horas) / 1000  
        
        fig_resp = go.Figure()
        fig_resp.add_trace(go.Scatter(x=horas, y=calor_acumulado, fill='tozeroy', mode='lines', 
                                      line=dict(color='#8d6e63', width=3), name="Energía"))
        
        fig_resp.update_layout(title=f"Energía térmica acumulada en 24h ({producto_sel.split()[0]})",
                               xaxis_title="Tiempo (Horas)", yaxis_title="Calor liberado (MJ)",
                               hovermode="x unified", height=300,
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_resp, use_container_width=True)
