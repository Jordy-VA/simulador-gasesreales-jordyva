import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve

# CONFIGURACIÓN
st.set_page_config(layout="wide", page_title="Termodinámica Agroindustrial", page_icon="🌱")

# CSS: Diseño de software industrial (Cajas de métricas verdes)
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
div[data-testid="stMetricValue"] > div {
    color: #81c784;
}
</style>
""", unsafe_allow_html=True)

# Base de datos
GASES_PROPS = {
    "Dióxido de Carbono (CO2)": (304.2, 73.8, 0.224),
    "Agua (H2O)":               (647.1, 220.6, 0.344),
    "Nitrógeno (N2)":           (126.2, 34.0, 0.037),
    "Etanol (C2H5OH)":          (513.9, 61.4, 0.644),
    "Metano (CH4)":             (190.6, 46.1, 0.011),
    "Amoníaco (NH3)":           (405.4, 113.5, 0.253), 
    "Propano (C3H8)":           (369.8, 42.5, 0.152)
}

R = 0.08314 

def parametros_eos(Tc, Pc, w, modelo="vdw"):
    if modelo == "vdw":
        return (27 * (R * Tc)**2) / (64 * Pc), (R * Tc) / (8 * Pc)
    else:
        a = (0.45724 * (R * Tc)**2) / Pc
        b = (0.07780 * R * Tc) / Pc
        kappa = 0.37464 + 1.54226 * w - 0.26992 * w**2
        return a, b, kappa

def maxwell_vdw(T, a, b):
    def sistema(vars):
        Vl, Vv, Psat = vars
        P_vl = (R * T) / (Vl - b) - a / (Vl**2)
        P_vv = (R * T) / (Vv - b) - a / (Vv**2)
        int_vdw = R * T * np.log((Vv - b) / (Vl - b)) + a * (1/Vv - 1/Vl)
        return [P_vl - Psat, P_vv - Psat, int_vdw - Psat * (Vv - Vl)]
    
    Vl_g = b * 1.5
    Vv_g = (R * T) / (0.5 * a / b**2) + b 
    Psat_g = (R * T) / (Vv_g - b) - a / (Vv_g**2)
    try:
        sol = fsolve(sistema, [Vl_g, Vv_g, max(Psat_g, 0.1)])
        return sol[0], sol[1], sol[2]
    except:
        return np.nan, np.nan, np.nan

# ==========================================
# INTERFAZ 
# ==========================================
st.title("🌱 Dashboard Termodinámico: Fluidos Reales")
st.markdown("Herramienta interactiva para ingeniería agroindustrial.")
st.markdown("---")

st.sidebar.header("⚙️ Panel de Control")
gas = st.sidebar.selectbox("Selecciona el Fluido de Trabajo:", list(GASES_PROPS.keys()))
modelo = st.sidebar.selectbox("Ecuación de Estado:", ["Van der Waals", "Peng-Robinson"])

st.sidebar.markdown("---")
st.sidebar.success("💡 **Tip Agroindustrial:**\nConocer el punto crítico del CO2 es vital para extraer aceites esenciales sin usar solventes tóxicos. El Amoníaco (NH3) es el rey de la refrigeración en la cadena de frío alimentaria.")

modelo_key = "vdw" if modelo == "Van der Waals" else "pr"
Tc, Pc, w = GASES_PROPS[gas]
Vc = (3 * R * Tc) / (8 * Pc) if modelo_key == "vdw" else 0.307 * R * Tc / Pc

if modelo_key == "vdw":
    a, b = parametros_eos(Tc, Pc, w, "vdw")
else:
    a, b, kappa = parametros_eos(Tc, Pc, w, "pr")

def calc_P(V, T):
    if modelo_key == "vdw":
        return (R * T) / (V - b) - (a / V**2)
    else:
        alpha = (1 + kappa * (1 - np.sqrt(T / Tc)))**2
        return (R * T) / (V - b) - (a * alpha) / (V**2 + 2*b*V - b**2)

# Tarjetas superiores de Métricas
col1, col2, col3 = st.columns(3)
col1.metric("Temperatura Crítica (Tc)", f"{Tc} K")
col2.metric("Presión Crítica (Pc)", f"{Pc} bar")
col3.metric("Volumen Crítico (Vc)", f"{Vc:.4f} L/mol")
st.markdown("<br>", unsafe_allow_html=True)

# Gráficos con PLOTLY (Transparentes e interactivos)
tab1, tab2 = st.tabs(["📉 Análisis 2D (Interactivo)", "🧊 Superficie 3D"])

with tab1:
    colA, colB = st.columns(2)
    
    V_arr = np.linspace(1.05 * b, 8 * Vc, 500)
    T_sub = Tc * 0.85
    T_crit = Tc
    T_sup = Tc * 1.15
    P_sub = calc_P(V_arr, T_sub)

    # 1. Gráfico P-V
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=V_arr, y=P_sub, name=f'Subcrítica ({T_sub:.1f} K)', line=dict(color='#4caf50', width=3)))
    
    if modelo_key == "vdw":
        Vl, Vv, Psat = maxwell_vdw(T_sub, a, b)
        if not np.isnan(Psat):
            fig1.add_trace(go.Scatter(x=[Vl, Vv], y=[Psat, Psat], name='Condensación', line=dict(color='gray', width=2, dash='dash')))
            
    fig1.add_trace(go.Scatter(x=V_arr, y=calc_P(V_arr, T_crit), name=f'Crítica ({T_crit:.1f} K)', line=dict(color='#ff9800', width=3)))
    fig1.add_trace(go.Scatter(x=V_arr, y=calc_P(V_arr, T_sup), name=f'Supercrítica ({T_sup:.1f} K)', line=dict(color='#ffeb3b', width=3)))
    fig1.add_trace(go.Scatter(x=[Vc], y=[Pc], mode='markers', name='Punto Crítico', marker=dict(color='red', size=10)))
    
    fig1.update_layout(title="Diagrama Presión vs Volumen", xaxis_title="Volumen (L/mol)", yaxis_title="Presión (bar)", 
                       yaxis=dict(range=[0, Pc * 2]), xaxis=dict(range=[0, Vc * 5]), hovermode="x unified")
    
    with colA:
        st.plotly_chart(fig1, use_container_width=True)

    # 2. Gráfico Z
    fig2 = go.Figure()
    Z_sub = (P_sub * V_arr) / (R * T_sub)
    Z_crit = (calc_P(V_arr, T_crit) * V_arr) / (R * T_crit)
    
    fig2.add_trace(go.Scatter(x=P_sub, y=Z_sub, name=f'T = {T_sub:.1f} K', line=dict(color='#4caf50', width=3)))
    fig2.add_trace(go.Scatter(x=calc_P(V_arr, T_crit), y=Z_crit, name=f'T = {T_crit:.1f} K', line=dict(color='#ff9800', width=3)))
    fig2.add_shape(type="line", x0=0, y0=1, x1=Pc*2, y1=1, line=dict(color="gray", width=2, dash="dash"))
    
    fig2.update_layout(title="Factor de Compresibilidad (Z)", xaxis_title="Presión (bar)", yaxis_title="Z", 
                       yaxis=dict(range=[0, 1.2]), xaxis=dict(range=[0, Pc * 2]), hovermode="x unified")
    
    with colB:
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown("Visualización espacial interactiva. ¡Gira y acerca la campana con el ratón o el dedo!")
    
    V_grid = np.linspace(1.2 * b, 5 * Vc, 50)
    T_grid = np.linspace(Tc * 0.7, Tc * 1.3, 50)
    V_mesh, T_mesh = np.meshgrid(V_grid, T_grid)
    P_mesh = calc_P(V_mesh, T_mesh)
    P_mesh = np.clip(P_mesh, 0, Pc * 3)
    
    fig3 = go.Figure(data=[go.Surface(z=P_mesh, x=V_mesh, y=T_mesh, colorscale='Greens', opacity=0.9)])
    fig3.add_trace(go.Scatter3d(x=[Vc], y=[Tc], z=[Pc], mode='markers', name='Punto Crítico', marker=dict(color='red', size=8)))
    
    fig3.update_layout(scene=dict(xaxis_title='Volumen', yaxis_title='Temperatura', zaxis_title='Presión'), margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig3, use_container_width=True)
