import streamlit as st
import numpy as np
import sympy
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="FSAE Suspension Analyzer", layout="wide")

st.title("🏎️ FSAE Suspension Stress & Geometry Analyzer")
st.markdown("Ajusta los puntos y parámetros en la izquierda para ver el análisis en tiempo real.")

# --- BARRA LATERAL (INPUTS) ---
with st.sidebar:
    st.header("1. Masas y Distribución")
    m_vehiculo = st.number_input("Masa Vehículo (kg)", value=300.0)
    m_piloto = st.number_input("Masa Piloto (kg)", value=65.0)
    weight_dist = st.slider("Distribución Delantera (%)", 0, 100, 40) / 100
    
    st.header("2. Geometría (Hardpoints)")
    st.info("Coordenadas en Metros [X, Y, Z]")
    
    # Función para crear inputs de puntos
    def input_point(label, default):
        cols = st.columns(3)
        x = cols[0].number_input(f"{label} X", value=default[0], format="%.4f")
        y = cols[1].number_input(f"{label} Y", value=default[1], format="%.4f")
        z = cols[2].number_input(f"{label} Z", value=default[2], format="%.4f")
        return [x, y, z]

    with st.expander("Rueda y Chasis"):
        tire_FL = input_point("Centro Rueda", [0.0, 0.5867, 0.2525])
        lwb_ch_f = input_point("LWB Chasis Front", [-0.0799, 0.2645, 0.1252])
        lwb_ch_r = input_point("LWB Chasis Rear", [0.0799, 0.2645, 0.1252])
        uwb_ch_f = input_point("UWB Chasis Front", [-0.0543, 0.2425, 0.2888])
        uwb_ch_r = input_point("UWB Chasis Rear", [0.0930, 0.2425, 0.2728])
        
    with st.expander("Upright y Otros"):
        lwb_up = input_point("LWB Upright (LBJ)", [0.0, 0.5843, 0.1485])
        uwb_up = input_point("UWB Upright (UBJ)", [0.0301, 0.5538, 0.3565])
        push_ch = input_point("Pushrod Chasis", [0.0, 0.2000, 0.5000])
        push_up = input_point("Pushrod Upright", [0.0, 0.5000, 0.2500])
        tie_ch = input_point("Tie-Rod Chasis", [-0.100, 0.270, 0.160])
        tie_up = input_point("Tie-Rod Upright", [-0.100, 0.520, 0.160])

    st.header("3. Material y Tubos")
    yield_str = st.number_input("Límite Elástico (MPa)", value=310.0)
    young_mod = st.number_input("Módulo Young (GPa)", value=210.0)
    od_mm = st.number_input("Diámetro Ext (mm)", value=20.0)
    id_mm = st.number_input("Diámetro Int (mm)", value=0.0)
    sf_target = st.number_input("FS Objetivo", value=3.5)

    st.header("4. Dinámica (G's)")
    g_brake = st.slider("G's Frenado", 0.0, 3.0, 1.7)
    g_corn = st.slider("G's Curva", 0.0, 3.0, 2.0)
    g_bump = st.slider("G's Bump", 1.0, 5.0, 3.0)
    h_cg = st.number_input("Altura CG (m)", value=0.35)
    w_base = st.number_input("Wheelbase (m)", value=1.65)
    track_w = st.number_input("Track Width (m)", value=1.20)

# --- LÓGICA DE CÁLCULO (Basada en tu Colab) ---

def solve_suspension(is_dynamic=False):
    g = 9.81
    total_mass = m_vehiculo + m_piloto
    
    # Cargas
    if not is_dynamic:
        # Estático simple
        susp_mass_front_corner = (total_mass * weight_dist) / 2
        fz_ground = susp_mass_front_corner * g
        fx_ground = 0
        fy_ground = 0
    else:
        # Dinámico con transferencia
        trans_long = (total_mass * g_brake * h_cg) / w_base
        axle_front_dyn = (total_mass * weight_dist) + trans_long
        trans_lat = (axle_front_dyn * g_corn * h_cg) / track_w
        mass_fl_dyn = (axle_front_dyn / 2) + trans_lat
        fz_ground = mass_fl_dyn * g * g_bump
        fx_ground = -min(mass_fl_dyn * g * g_brake, fz_ground) # Simplificado
        fy_ground = min(mass_fl_dyn * g * g_corn, fz_ground)

    # Solver Simbólico
    f_lwbf, f_lwbr, f_uwbf, f_uwbr, f_push, f_tie = sympy.symbols('f_lwbf f_lwbr f_uwbf f_uwbr f_push f_tie')
    
    def unit_v(p1, p2):
        v = np.array(p2) - np.array(p1)
        return v / np.linalg.norm(v)

    # Vectores
    v_lwbf = unit_v(lwb_ch_f, lwb_up); v_lwbr = unit_v(lwb_ch_r, lwb_up)
    v_uwbf = unit_v(uwb_ch_f, uwb_up); v_uwbr = unit_v(uwb_ch_r, uwb_up)
    v_push = unit_v(push_ch, push_up); v_tie = unit_v(tie_ch, tie_up)

    # Ecuaciones
    f_vecs = [
        f_lwbf * sympy.Matrix(v_lwbf), f_lwbr * sympy.Matrix(v_lwbr),
        f_uwbf * sympy.Matrix(v_uwbf), f_uwbr * sympy.Matrix(v_uwbr),
        f_push * sympy.Matrix(v_push), f_tie * sympy.Matrix(v_tie)
    ]
    
    sum_f = sympy.Matrix([fx_ground, fy_ground, fz_ground]) + sum(f_vecs, sympy.Matrix([0,0,0]))
    
    # Momentos respecto al centro de rueda
    ref = np.array(tire_FL)
    r_ground = sympy.Matrix([tire_FL[0], tire_FL[1], 0.0] - ref)
    r_lwb = sympy.Matrix(np.array(lwb_up) - ref)
    r_uwb = sympy.Matrix(np.array(uwb_up) - ref)
    r_push = sympy.Matrix(np.array(push_up) - ref)
    r_tie = sympy.Matrix(np.array(tie_up) - ref)

    m_ground = r_ground.cross(sympy.Matrix([fx_ground, fy_ground, fz_ground]))
    m_total = m_ground + r_lwb.cross(f_vecs[0]) + r_lwb.cross(f_vecs[1]) + \
              r_uwb.cross(f_vecs[2]) + r_uwb.cross(f_vecs[3]) + \
              r_push.cross(f_vecs[4]) + r_tie.cross(f_vecs[5])

    eqs = [sympy.Eq(sum_f[i], 0) for i in range(3)] + [sympy.Eq(m_total[i], 0) for i in range(3)]
    sol = sympy.solve(eqs, [f_lwbf, f_lwbr, f_uwbf, f_uwbr, f_push, f_tie])
    
    return sol, {
        "LWB F": (sol[f_lwbf], lwb_ch_f, lwb_up),
        "LWB R": (sol[f_lwbr], lwb_ch_r, lwb_up),
        "UWB F": (sol[f_uwbf], uwb_ch_f, uwb_up),
        "UWB R": (sol[f_uwbr], uwb_ch_r, uwb_up),
        "Pushrod": (sol[f_push], push_ch, push_up),
        "Tie-Rod": (sol[f_tie], tie_ch, tie_up)
    }

# --- INTERFAZ DE RESULTADOS ---
col_viz, col_res = st.columns([2, 1])

with col_res:
    mode = st.radio("Cargar Caso:", ["Estático", "Dinámico"])
    is_dyn = (mode == "Dinámico")
    
    try:
        solution, members = solve_suspension(is_dynamic=is_dyn)
        
        st.subheader("Fuerzas y Esfuerzos")
        area = (np.pi / 4) * (od_mm**2 - id_mm**2)
        inertia = (np.pi / 64) * (od_mm**4 - id_mm**4)
        
        results_data = []
        for name, (force, p1, p2) in members.items():
            f_val = float(force)
            length_mm = np.linalg.norm(np.array(p2)-np.array(p1)) * 1000
            stress = abs(f_val) / area
            fs = yield_str / stress if stress > 0 else 999
            
            # Pandeo
            buckling = "N/A"
            if f_val > 0: # Compresión
                p_crit = (np.pi**2 * (young_mod*1000) * inertia) / (length_mm**2)
                fs_b = p_crit / f_val
                buckling = f"{fs_b:.2f}"

            results_data.append({
                "Miembro": name,
                "Fuerza [N]": round(f_val, 1),
                "Stress [MPa]": round(stress, 1),
                "FS Fluencia": round(fs, 2),
                "FS Pandeo": buckling
            })
        
        st.table(results_data)
        
    except Exception as e:
        st.error(f"Error en el cálculo: {e}. Revisa que los puntos no sean coplanares.")

with col_viz:
    st.subheader("Visualización 3D")
    
    fig = go.Figure()

    # Función para dibujar barras
    def draw_link(p1, p2, name, color, width=5):
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines+markers',
            line=dict(color=color, width=width),
            marker=dict(size=3, color='black'),
            name=name
        ))

    # Dibujar Suspensión
    draw_link(lwb_ch_f, lwb_up, "LWB Front", 'blue')
    draw_link(lwb_ch_r, lwb_up, "LWB Rear", 'blue')
    draw_link(uwb_ch_f, uwb_up, "UWB Front", 'red')
    draw_link(uwb_ch_r, uwb_up, "UWB Rear", 'red')
    draw_link(push_ch, push_up, "Pushrod", 'green', width=8)
    draw_link(tie_ch, tie_up, "Tie-Rod", 'orange')
    
    # Upright (Línea entre UBJ y LBJ)
    draw_link(lwb_up, uwb_up, "Upright", 'black', width=10)

    # Rueda (Simplificada como un círculo o punto)
    fig.add_trace(go.Scatter3d(
        x=[tire_FL[0]], y=[tire_FL[1]], z=[tire_FL[2]],
        mode='markers', marker=dict(size=10, color='black'), name='Rueda'
    ))

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title="X (Longitudinal)",
            yaxis_title="Y (Transversal)",
            zaxis_title="Z (Vertical)"
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)