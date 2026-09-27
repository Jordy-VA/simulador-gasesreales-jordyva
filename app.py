import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(layout="wide", page_title="Termodinámica Agroindustrial", page_icon="🌱")

# Estilo visual de las gráficas
plt.style.use('ggplot')

# Base de datos: Tc [K], Pc [bar], w (factor acéntrico)
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
        a = (27 * (R * Tc)**2) / (64 * Pc)
        b = (R * Tc) / (8 * Pc)
        return a, b
    elif modelo == "pr":
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
# INTERFAZ DE USUARIO (UI)
# ==========================================

st.title("🌱 Dashboard Termodinámico: Fluidos Reales")
st.markdown("Herramienta de análisis para procesos de refrigeración y fluidos supercríticos en la agroindustria.")
st.markdown("---")

st.sidebar.header("⚙️ Panel de Control")
gas = st.sidebar.selectbox("Selecciona el Fluido de Trabajo:", list(GASES_PROPS.keys()))
modelo = st.sidebar.selectbox("Ecuación de Estado:", ["Van der Waals", "Peng-Robinson"])

st.sidebar.markdown("---")
st.sidebar.success("💡 **Tip Agroindustrial:**\nConocer el punto crítico del CO2 es vital para extraer aceites esenciales sin degradarlos térmicamente. Fluidos como el Amoníaco (NH3) son fundamentales en los sistemas de refrigeración de la cadena de frío.")

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

st.subheader(f"📊 Propiedades Críticas del {gas.split()[0]}")
col1, col2, col3 = st.columns(3)
col1.metric(label="Temperatura Crítica (Tc)", value=f"{Tc} K")
col2.metric(label="Presión Crítica (Pc)", value=f"{Pc} bar")
col3.metric(label="Volumen Crítico (Vc)", value=f"{Vc:.4f} L/mol")
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📉 Análisis 2D (Isotermas y Factor Z)", "🧊 Superficie 3D (P-V-T)"])

with tab1:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('none') 
    
    V_arr = np.linspace(1.05 * b, 8 * Vc, 500)
    T_sub = Tc * 0.85
    T_crit = Tc
    T_sup = Tc * 1.15

    P_sub = calc_P(V_arr, T_sub)
    
    # NUEVA PALETA AGRO: Verde hoja, Terracota, Amarillo trigo
    ax1.plot(V_arr, P_sub, color='#2E7D32', label=f'Subcrítica ({T_sub:.1f} K)')
    
    if modelo_key == "vdw":
        Vl, Vv, Psat = maxwell_vdw(T_sub, a, b)
        if not np.isnan(Psat):
            ax1.plot([Vl, Vv], [Psat, Psat], 'k--', lw=2, label='Condensación (Maxwell)')
            ax1.scatter([Vl, Vv], [Psat, Psat], color='black', zorder=5)
            V_mid = V_arr[(V_arr >= Vl) & (V_arr <= Vv)]
            P_mid = P_sub[(V_arr >= Vl) & (V_arr <= Vv)]
            ax1.fill_between(V_mid, Psat, P_mid, where=(P_mid > Psat), color='#FFAB91', alpha=0.3)
            ax1.fill_between(V_mid, Psat, P_mid, where=(P_mid < Psat), color='#81C784', alpha=0.3)

    ax1.plot(V_arr, calc_P(V_arr, T_crit), color='#D84315', lw=2.5, label=f'Crítica ({T_crit:.1f} K)')
    ax1.plot(V_arr, calc_P(V_arr, T_sup), color='#FBC02D', label=f'Supercrítica ({T_sup:.1f} K)')
    ax1.scatter([Vc], [Pc], color='#3E2723', s=80, zorder=6, label='Punto Crítico')
    
    ax1.set_title(f'Diagrama P-V', fontweight='bold')
    ax1.set_xlabel('Volumen (L/mol)')
    ax1.set_ylabel('Presión (bar)')
    ax1.set_ylim(0, Pc * 2)
    ax1.set_xlim(0, Vc * 5)
    ax1.legend()

    Z_sub = (P_sub * V_arr) / (R * T_sub)
    Z_crit = (calc_P(V_arr, T_crit) * V_arr) / (R * T_crit)
    ax2.plot(P_sub, Z_sub, color='#2E7D32', label=f'T = {T_sub:.1f} K')
    ax2.plot(calc_P(V_arr, T_crit), Z_crit, color='#D84315', lw=2.5, label=f'T = {T_crit:.1f} K')
    ax2.axhline(1, color='gray', linestyle='--', label='Gas Ideal (Z=1)')
    ax2.set_title(f'Desviación de la Idealidad (Z)', fontweight='bold')
    ax2.set_xlabel('Presión (bar)')
    ax2.set_ylabel('Factor Z')
    ax2.set_xlim(0, Pc * 2)
    ax2.set_ylim(0, 1.2)
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.markdown("Visualización espacial de la cúpula de saturación. El punto central oscuro marca el estado crítico.")
    fig3d = plt.figure(figsize=(10, 7))
    fig3d.patch.set_facecolor('none')
    ax3d = fig3d.add_subplot(111, projection='3d')
    
    V_grid = np.linspace(1.2 * b, 5 * Vc, 50)
    T_grid = np.linspace(Tc * 0.7, Tc * 1.3, 50)
    V_mesh, T_mesh = np.meshgrid(V_grid, T_grid)
    P_mesh = calc_P(V_mesh, T_mesh)
    P_mesh = np.clip(P_mesh, 0, Pc * 3)
    
    # NUEVA PALETA 3D: YlGn (Amarillos y Verdes)
    surf = ax3d.plot_surface(V_mesh, T_mesh, P_mesh, cmap='YlGn', edgecolor='none', alpha=0.9)
    ax3d.scatter([Vc], [Tc], [Pc], color='#3E2723', s=100, label='Punto Crítico')
    ax3d.set_title(f'Superficie 3D Termodinámica', fontweight='bold')
    ax3d.set_xlabel('Volumen (L/mol)')
    ax3d.set_ylabel('Temperatura (K)')
    ax3d.set_zlabel('Presión (bar)')
    
    st.pyplot(fig3d)
