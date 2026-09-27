import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Base de datos
GASES_PROPS = {
    "Dióxido de Carbono": (304.2, 73.8),
    "Agua (H2O)":         (647.1, 220.6),
    "Nitrógeno (N2)":     (126.2, 34.0),
    "Etanol (C2H5OH)":    (513.9, 61.4),
    "Metano (CH4)":       (190.6, 46.1),
    "Propano (C3H8)":     (369.8, 42.5)
}

R = 0.08314 

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

# Interfaz web
st.title("Simulador Termodinámico de Gases Reales")
st.write("Selecciona un gas para calcular su punto crítico y graficar sus isotermas.")

gas_seleccionado = st.selectbox("Elige un gas:", list(GASES_PROPS.keys()))

Tc, Pc = GASES_PROPS[gas_seleccionado]
a = (27 * (R * Tc)**2) / (64 * Pc)
b = (R * Tc) / (8 * Pc)
Vc = (3 * R * Tc) / (8 * Pc)

# Crear gráfico
fig, ax = plt.subplots(figsize=(8, 5))
V_arr = np.linspace(1.05 * b, 8 * Vc, 500)
T_sub = Tc * 0.85
T_crit = Tc

# Cálculos y ploteo
P_sub = (R * T_sub) / (V_arr - b) - (a / V_arr**2)
ax.plot(V_arr, P_sub, 'royalblue', label=f'Subcrítica ({T_sub:.1f} K)')

Vl, Vv, Psat = maxwell_vdw(T_sub, a, b)
if not np.isnan(Psat):
    ax.plot([Vl, Vv], [Psat, Psat], 'k--', lw=2, label='Cambio de Fase')
    ax.scatter([Vl, Vv], [Psat, Psat], color='black', zorder=5)

P_crit = (R * T_crit) / (V_arr - b) - (a / V_arr**2)
ax.plot(V_arr, P_crit, 'r-', lw=2.5, label=f'Crítica ({T_crit:.1f} K)')
ax.scatter([Vc], [Pc], color='darkred', s=80, label='Punto Crítico', zorder=5)

ax.set_title(f"Isotermas de Van der Waals: {gas_seleccionado}")
ax.set_xlabel("Volumen Molar (L/mol)")
ax.set_ylabel("Presión (bar)")
ax.set_ylim(0, Pc * 2)
ax.set_xlim(0, Vc * 4)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# Mostrar gráfico en la web
st.pyplot(fig)

st.write(f"**Propiedades Críticas calculadas:** $T_c = {Tc} K$, $P_c = {Pc} bar$, $V_c = {Vc:.4f} L/mol$")
