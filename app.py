import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve

# ==========================================
# 1. CONFIGURACIÓN Y ESTILOS
# ==========================================
st.set_page_config(layout="wide", page_title="Termodinámica Agroindustrial", page_icon="🌱")

st.markdown("""
<style>
div[data-testid="metric-container"] {
    background-color: #1a241b;
    border: 1px solid #2e7d32;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #4caf50;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
}
div[data-testid="stMetricValue"] > div { color: #81c784; }
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

# ==========================================
# 3. INTERFAZ DE USUARIO (UI)
# ==========================================
st.title("🌱 Dashboard Termodinámico: Fluidos Reales y Cinética Molecular")
st.markdown("Herramienta interactiva para ingeniería agroindustrial y fisicoquímica.")

st.sidebar.header("⚙️ Panel de Control")

gas = st.sidebar.selectbox("Selecciona el Fluido de Trabajo:", list(GASES_PROPS.keys()))
modelo = st.sidebar.selectbox("Ecuación de Estado:", ["Van der Waals", "Peng-Robinson"])

st.sidebar.markdown("---")

TIPS_AGRO = {
    "Dióxido de Carbono (CO2)": "💡 **Aplicación Agroindustrial:** El CO2 supercrítico es vital para extraer aceites esenciales o descafeinar café. Su baja temperatura crítica (31°C) evita la degradación térmica de compuestos sensibles.",
    "Agua (H2O)": "💡 **Aplicación Agroindustrial:** El vapor sobrecalentado es el principal medio de transferencia de calor en marmitas, pasteurización y esterilización de alimentos.",
    "Nitrógeno (N2)": "💡 **Aplicación Agroindustrial:** En fase líquida (criogénico) se usa para el congelamiento ultra rápido (IQF) de frutas y hortalizas. Como gas, se usa en envasado de atmósfera modificada.",
    "Etanol (C2H5OH)": "💡 **Aplicación Agroindustrial:** Solvente orgánico GRAS (seguro) utilizado en la extracción de pigmentos y biocompuestos, y como fluido secundario en refrigeración.",
    "Metano (CH4)": "💡 **Aplicación Agroindustrial:** Componente principal del biogás, obtenido por biodigestión anaerobia de residuos agrícolas para la cogeneración de energía térmica y eléctrica.",
    "Amoníaco (NH3)": "💡 **Aplicación Agroindustrial:** El refrigerante industrial por excelencia. Su alto calor latente de vaporización lo hace el rey de la cadena de frío en frigoríficos de agroexportación.",
    "Propano (C3H8)": "💡 **Aplicación Agroindustrial:** Conocido como R-290, es un refrigerante ecológico que está reemplazando a los freones en equipos comerciales por su bajo potencial de calentamiento global."
}

st.sidebar.success(TIPS_AGRO[gas])

Tc, Pc, w, MasaMolar = GASES_PROPS[gas]
modelo_key = "vdw" if modelo == "Van der Waals" else "pr"
Vc = (3 * R_bar * Tc) / (8 * Pc) if modelo_key == "vdw" else 0.307 * R_bar * Tc / Pc

if modelo_key == "vdw":
    a, b = parametros_eos(Tc, Pc, w, "vdw")
else:
    a, b, kappa = parametros_eos(Tc, Pc, w, "pr")

def calc_P(V, T):
    if modelo_key == "vdw":
        return (R_bar * T) / (V - b) - (a / V**2)
    else:
        alpha = (1 + kappa * (1 - np.sqrt(T / Tc)))**2
        return (R_bar * T) / (V - b) - (a * alpha) / (V**2 + 2*b*V - b**2)

# Tarjetas superiores
col1, col2, col3 = st.columns(3)
col1.metric("Temperatura Crítica (Tc)", f"{Tc} K")
col2.metric("Presión Crítica (Pc)", f"{Pc} bar")
col3.metric("Masa Molar (M)", f"{MasaMolar*1000:.2f} g/mol")
st.markdown("<br>", unsafe_allow_html=True)

# Pestañas
tab1, tab2, tab3 = st.tabs(["📉 Análisis 2D (Gases Reales)", "🧊 Superficie 3D", "🚀 Velocidad Molecular (Maxwell)"])

# --- PESTAÑA 1: Gráficos 2D ---
with tab1:
    colA, colB = st.columns(2)
    V_arr = np.linspace(1.05 * b, 8 * Vc, 500)
    T_sub, T_crit, T_sup = Tc * 0.85, Tc, Tc * 1.15
    P_sub = calc_P(V_arr, T_sub)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=V_arr, y=P_sub, name=f'Subcrítica ({T_sub:.1f} K)', line=dict(color='#4caf50', width=3)))
    if modelo_key == "vdw":
        Vl, Vv, Psat = maxwell_vdw(T_sub, a, b)
        if not np.isnan(Psat):
            fig1.add_trace(go.Scatter(x=[Vl, Vv], y=[Psat, Psat], name='Condensación (Maxwell)', line=dict(color='gray', width=2, dash='dash')))
    fig1.add_trace(go.Scatter(x=V_arr, y=calc_P(V_arr, T_crit), name=f'Crítica ({T_crit:.1f} K)', line=dict(color='#ff9800', width=3)))
    fig1.add_trace(go.Scatter(x=V_arr, y=calc_P(V_arr, T_sup), name=f'Supercrítica ({T_sup:.1f} K)', line=dict(color='#ffeb3b', width=3)))
    fig1.add_trace(go.Scatter(x=[Vc], y=[Pc], mode='markers', name='Punto Crítico', marker=dict(color='red', size=10)))
    
    fig1.update_layout(title="Diagrama Presión vs Volumen", xaxis_title="Volumen (L/mol)", yaxis_title="Presión (bar)", 
                       yaxis=dict(range=[0, Pc * 2]), xaxis=dict(range=[0, Vc * 5]), hovermode="x unified")
    with colA: 
        st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=P_sub, y=(P_sub * V_arr) / (R_bar * T_sub), name=f'T = {T_sub:.1f} K', line=dict(color='#4caf50', width=3)))
    fig2.add_trace(go.Scatter(x=calc_P(V_arr, T_crit), y=(calc_P(V_arr, T_crit) * V_arr) / (R_bar * T_crit), name=f'T = {T_crit:.1f} K', line=dict(color='#ff9800', width=3)))
    fig2.add_shape(type="line", x0=0, y0=1, x1=Pc*2, y1=1, line=dict(color="gray", width=2, dash="dash"))
    
    fig2.update_layout(title="Factor de Compresibilidad (Z)", xaxis_title="Presión (bar)", yaxis_title="Z", 
                       yaxis=dict(range=[0, 1.2]), xaxis=dict(range=[0, Pc * 2]), hovermode="x unified")
    with colB: 
        st.plotly_chart(fig2, use_container_width=True)

# --- PESTAÑA 2: Superficie 3D ---
with tab2:
    st.markdown("Visualización espacial interactiva. ¡Gira y acerca la campana P-V-T con el ratón!")
    
    V_grid, T_grid = np.meshgrid(np.linspace(1.2 * b, 5 * Vc, 50), np.linspace(Tc * 0.7, Tc * 1.3, 50))
    P_mesh = np.clip(calc_P(V_grid, T_grid), 0, Pc * 3)
    
    fig3 = go.Figure(data=[go.Surface(z=P_mesh, x=V_grid, y=T_grid, colorscale='Greens', opacity=0.9)])
    fig3.add_trace(go.Scatter3d(x=[Vc], y=[Tc], z=[Pc], mode='markers', name='Punto Crítico', marker=dict(color='red', size=8)))
    
    fig3.update_layout(scene=dict(xaxis_title='Volumen', yaxis_title='Temperatura', zaxis_title='Presión'), margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig3, use_container_width=True)

# --- PESTAÑA 3: Maxwell-Boltzmann ---
with tab3:
    st.markdown("### Distribución de Velocidades de Maxwell-Boltzmann")
    st.write(f"Cinética molecular del **{gas}** en función de la temperatura.")
    
    temp_mb = st.slider("🌡️ Modifica la Temperatura del Gas (Kelvin):", min_value=100, max_value=1500, value=int(Tc), step=50)
    
    vp = np.sqrt((2 * R_J * temp_mb) / MasaMolar)          
    v_prom = np.sqrt((8 * R_J * temp_mb) / (np.pi * MasaMolar)) 
    v_rms = np.sqrt((3 * R_J * temp_mb) / MasaMolar)       
    
    v = np.linspace(0, v_rms * 3, 1000)
    fv = 4 * np.pi * (MasaMolar / (2 * np.pi * R_J * temp_mb))**(1.5) * (v**2) * np.exp(-MasaMolar * (v**2) / (2 * R_J * temp_mb))
    
    fig_mb = go.Figure()
    fig_mb.add_trace(go.Scatter(x=v, y=fv, fill='tozeroy', mode='lines', line=dict(color='#00bcd4', width=3), name="Distribución"))
    
    fig_mb.add_vline(x=vp, line_dash="dash", line_color="#fbc02d", 
                     annotation_text=f"Más probable ({vp:.0f} m/s)", 
                     annotation_position="top left")
                     
    fig_mb.add_vline(x=v_rms, line_dash="dash", line_color="#d32f2f", 
                     annotation_text=f"RMS ({v_rms:.0f} m/s)", 
                     annotation_position="top right")
    
    fig_mb.update_layout(
        xaxis_title="Velocidad de las moléculas (metros/segundo)",
        yaxis_title="Probabilidad",
        hovermode="x unified",
        showlegend=False
    )
    
    st.plotly_chart(fig_mb, use_container_width=True)

# --- PESTAÑA 4: Simulador de Enfriamiento Agroindustrial ---
with tab4:
    st.header("❄️ Simulador de Enfriamiento y Equilibrio Térmico")
    st.markdown("Proyección de tiempo y energía para alcanzar el equilibrio térmico en cámaras de frío.")
    
    col_input, col_grafico = st.columns([1, 2])
    
    with col_input:
        st.subheader("Parámetros del Sistema")
        
        # Productos basados en los ejercicios de la clase
        producto = st.selectbox("Selecciona el Producto:", ["Néctar de Maracuyá", "Leche Cruda", "Pulpa de Mango"])
        
        # Asignar propiedades intensivas (Densidad y Calor Específico aprox)
        if producto == "Néctar de Maracuyá":
            densidad = 1.05  # kg/L
            cp = 3.8         # kJ/kg°C
            t_ini_default = 25.0
            t_camara_default = 5.0
            k_enfriamiento = 0.02 # Constante de enfriamiento
        elif producto == "Leche Cruda":
            densidad = 1.03  # kg/L basado en el PDF
            cp = 3.93
            t_ini_default = 30.0
            t_camara_default = 4.0
            k_enfriamiento = 0.015
        else:
            densidad = 1.01
            cp = 3.6
            t_ini_default = 20.0
            t_camara_default = -18.0
            k_enfriamiento = 0.025
            
        # Propiedad Extensiva: Volumen
        volumen = st.number_input("Volumen del lote (Litros):", min_value=10, max_value=5000, value=1000, step=100)
        
        st.markdown("---")
        t_inicial = st.slider("Temp. Inicial del Producto (°C):", -10.0, 90.0, t_ini_default)
        t_camara = st.slider("Temp. de la Cámara/Entorno (°C):", -25.0, 30.0, t_camara_default)
        
        # Cálculos termodinámicos
        masa = volumen * densidad
        calor_remover = masa * cp * (t_inicial - t_camara)
        
    with col_grafico:
        # Ley de Enfriamiento de Newton para simular el proceso hacia el equilibrio
        tiempo = np.linspace(0, 300, 200) # Simulación de 0 a 300 minutos
        temperatura_t = t_camara + (t_inicial - t_camara) * np.exp(-k_enfriamiento * tiempo)
        
        fig_enfriamiento = go.Figure()
        
        # Curva de enfriamiento del producto
        fig_enfriamiento.add_trace(go.Scatter(x=tiempo, y=temperatura_t, mode='lines', 
                                              name=f'Temp. {producto}', line=dict(color='#ff9800', width=4)))
        
        # Línea de la cámara (Equilibrio)
        fig_enfriamiento.add_hline(y=t_camara, line_dash="dash", line_color="#00bcd4", 
                                   annotation_text=f"Equilibrio Térmico ({t_camara}°C)", annotation_position="bottom right")
        
        fig_enfriamiento.update_layout(
            title=f"Curva de Aproximación al Equilibrio Térmico ({producto})",
            xaxis_title="Tiempo (Minutos)",
            yaxis_title="Temperatura (°C)",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig_enfriamiento, use_container_width=True)

    # Panel de Resultados Termodinámicos
    st.markdown("### 📊 Reporte Termodinámico del Proceso")
    res1, res2, res3 = st.columns(3)
    
    res1.info(f"**Propiedad Extensiva (Masa):**\n\n{masa:,.2f} kg\n\n*(Calculado con densidad de {densidad} kg/L)*")
    res2.warning(f"**Transferencia de Calor (Q):**\n\n{calor_remover:,.2f} kJ\n\n*(Energía a extraer para llegar al equilibrio)*")
    
    # Conversión de temperatura final
    t_fin_k = t_camara + 273.15
    t_fin_f = (t_camara * 1.8) + 32
    res3.success(f"**Estado Final (Ley Cero):**\n\n{t_camara}°C | {t_fin_k} K | {t_fin_f}°F\n\n*(Sistema estabilizado)*")
