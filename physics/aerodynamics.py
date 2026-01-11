import numpy as np
from numba import njit

# --- Numba JIT Compiled Kernels ---

@njit(cache=True)
def get_airfoil_coeffs_jit(alpha_rad, polar_data):
    """
    Numba 优化的翼型插值
    polar_data: [N, 4] array (alpha_deg, cl, cd, cm)
    """
    alpha_deg = np.degrees(alpha_rad)
    # 归一化到 -180 ~ 180
    alpha_deg = (alpha_deg + 180) % 360 - 180
    
    # np.interp 在 numba 中是支持的
    cl = np.interp(alpha_deg, polar_data[:,0], polar_data[:,1])
    cd = np.interp(alpha_deg, polar_data[:,0], polar_data[:,2])
    # cm = np.interp(alpha_deg, polar_data[:,0], polar_data[:,3]) # 暂未使用 Cm
    return cl, cd

@njit(cache=True)
def solve_bem_jit(V_x, V_y, r, chord, twist, pitch_rad, polar_data, R_rotor, num_blades, rho):
    """
    BEM 核心迭代求解器 (JIT 加速版)
    """
    a = 0.0
    a_prime = 0.0
    phi = 0.0
    cn = 0.0
    ct = 0.0
    
    sigma = (num_blades * chord) / (2 * np.pi * r)
    
    # 迭代循环 (机器码级循环)
    for _ in range(10):
        # 1. Inflow Angle
        if abs(V_y) < 0.01: V_y = 0.01
        phi = np.arctan2(V_x * (1 - a), V_y * (1 + a_prime))
        
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        if abs(sin_phi) < 0.01: 
            sin_phi = 0.01 # Avoid div by zero
        
        # 2. AoA & Coeffs
        alpha_geom = phi - twist - pitch_rad
        cl, cd = get_airfoil_coeffs_jit(alpha_geom, polar_data)
        
        cn = cl * cos_phi + cd * sin_phi
        ct = cl * sin_phi - cd * cos_phi
        
        # 3. Prandtl Tip Loss
        f_tip = (num_blades / 2.0) * (R_rotor - r) / (r * abs(sin_phi))
        F = (2.0 / np.pi) * np.arccos(np.exp(-f_tip))
        if F < 0.001: F = 0.001
        
        # 4. Induction (Buhl Correction)
        coef = (4.0 * F * sin_phi**2) / (sigma * cn + 1e-8)
        a_classic = 1.0 / (coef + 1.0)
        
        if a_classic <= 0.4:
            a_new = a_classic
        else:
            # Buhl correction (Simplified stable form for high CT)
            # a_new = (18F - 20 - 3*sqrt(...)) / (36F - 50)
            # Robust fallback: Linear extrapolation or simple clamp
            a_new = 0.4 + 0.2 * (a_classic - 0.4) # Soft increase
            if a_new > 0.9: a_new = 0.9
            
        denom_ap = (4.0 * F * sin_phi * cos_phi) / (sigma * ct + 1e-8)
        ap_new = 1.0 / (denom_ap - 1.0)
        
        # Relaxation
        k = 0.3
        a = k * a_new + (1-k) * a
        a_prime = k * ap_new + (1-k) * a_prime
        
    #return phi, cn, ct
    return a, a_prime, phi, alpha_geom, cn, ct
 # 修改部分
    # === 返回更完整的 BEM 求解结果 ===
    # a: 轴向诱导因子
    # a_prime: 切向诱导因子
    # phi: 入流角
    # alpha_geom: 几何攻角（φ - 扭角 - 桨距）
    # cn, ct: 法向和切向力系数

@njit(cache=True)
def calculate_blade_forces_jit(
    num_blades, R_rotor, hub_height, rho,
    elem_r, elem_dr, elem_chord, elem_twist, polar_data,
    v_surge, v_pitch, rotor_speed, rotor_azimuth, 
    q_dot_flaps, v_wind_hub, pitch_cmd_rad
):
    """
    全叶片积分循环 (JIT 加速)
    Returns: 22-DOF Force Vector (Contribution)
    """
    F_gen = np.zeros(22)

 # ==== 新增：为诊断输出分配数组空间 ====#修改部分
    phi_out = np.zeros(len(elem_r))  # 入流角分布
    alpha_out = np.zeros(len(elem_r))  # 攻角分布
    a_out = np.zeros(len(elem_r))  # 轴向诱导因子
    ap_out = np.zeros(len(elem_r))  # 切向诱导因子
    cn_out = np.zeros(len(elem_r))  # 法向力系数
    ct_out = np.zeros(len(elem_r))  # 切向力系数


    total_thrust = 0.0
    total_torque = 0.0
    
    num_elems = len(elem_r)
    
    for b in range(num_blades):
        azimuth = rotor_azimuth + b * (2*np.pi/3)
        q_dot_flap1 = q_dot_flaps[b] # Flap1 velocity for this blade
        
        for i in range(num_elems):
            r = elem_r[i]
            dr = elem_dr[i]
            chord = elem_chord[i]
            twist = elem_twist[i]
            
            # --- Kinematics ---
            v_rot = rotor_speed * r
            v_axial_plat = v_surge + v_pitch * hub_height
            
            shape_f1 = (r / R_rotor)**2
            v_elastic_flap = q_dot_flap1 * shape_f1
            
            V_x = v_wind_hub - v_axial_plat - v_elastic_flap
            V_y = v_rot 
            
            # --- BEM Solve ---
            #phi, cn, ct = solve_bem_jit(
                #V_x, V_y, r, chord, twist, pitch_cmd_rad, 
                #polar_data, R_rotor, num_blades, rho
            #)

            # --- BEM Solve ---
            a, a_prime, phi, alpha_geom, cn, ct = solve_bem_jit(
                V_x, V_y, twist, pitch_cmd_rad,
                chord, dr, polar_data,
                rho, num_blades, R_rotor
            )

# === 保存诊断输出 ===#修改部分
            phi_out[i] = phi
            alpha_out[i] = alpha_geom
            a_out[i] = a
            ap_out[i] = a_prime
            cn_out[i] = cn
            ct_out[i] = ct

            # --- Forces ---
            W_sq = V_x**2 + V_y**2
            dyn_pres = 0.5 * rho * W_sq * chord * dr
            
            f_norm = dyn_pres * cn
            f_tang = dyn_pres * ct
            
            # --- Mapping ---
            # 1. Platform
            F_gen[0] += f_norm # Surge
            F_gen[4] += f_norm * hub_height # Pitch Moment
            
            # 2. Drivetrain
            torque = f_tang * r
            F_gen[11] += torque
            F_gen[12] -= torque
            
            # 3. Blade Elastic (Flap1)
            # F_gen index for Blade b Flap1: 13 + b*3
            F_gen[13 + b*3] += f_norm * shape_f1
            # Edge1 (simplified): 13 + b*3 + 1
            F_gen[13 + b*3 + 1] += f_tang * (r / R_rotor)
            
            total_thrust += f_norm
            total_torque += torque
            
    #return F_gen, total_thrust, total_torque

 # 返回结构力、风轮推力/扭矩，以及诊断变量#修改部分
        return (
            F_gen,
            total_thrust,
            total_torque,
            phi_out,  # 入流角(展向分布)
            alpha_out,  # 攻角(展向分布)
            a_out,  # 轴向诱导因子
            ap_out,  # 切向诱导因子
            cn_out,  # 法向力系数
            ct_out  # 切向力系数
        )


# --- Main Class ---

class Airfoil:
    def __init__(self, name="NREL-5MW"):
        self.name = name
        self.static_data = np.array([
            [-180, 0.0, 0.5, 0.0], [-20, -0.5, 0.1, -0.1], [-10, -0.8, 0.02, -0.05],
            [0, 0.0, 0.008, 0.0], [5, 0.6, 0.01, 0.02], [10, 1.1, 0.02, 0.05],
            [15, 1.3, 0.05, 0.08], [20, 1.0, 0.15, 0.10], [30, 0.8, 0.30, 0.15],
            [90, 0.0, 1.0, 0.0], [180, 0.0, 0.5, 0.0]
        ], dtype=np.float64)

class Aerodynamics:
    """
    WOUSE V4.3.2 气动模块 (Numba Accelerated)
    """
    def __init__(self, params):
        self.p = params
        self.rho = self.p.AirDensity
        self.last_thrust = 0.0
        self.last_torque = 0.0
        
        # Prepare Arrays for JIT (Structure of Arrays)
        self.r_nodes = np.array([2.8, 5.6, 8.3, 11.75, 15.85, 19.95, 24.05, 28.15, 
                            32.25, 36.35, 40.45, 44.55, 48.65, 52.75, 56.16, 58.9, 61.6])
        self.chord_nodes = np.array([3.5, 3.8, 4.1, 4.5, 4.6, 4.5, 4.2, 4.0, 
                                3.7, 3.5, 3.2, 3.0, 2.7, 2.5, 2.3, 2.1, 1.4])
        self.twist_nodes = np.radians([13.3, 13.3, 13.3, 13.3, 11.4, 10.1, 9.0, 7.7, 
                                  6.5, 5.3, 4.1, 3.1, 2.3, 1.5, 0.8, 0.3, 0.1])
        
        self.dr_nodes = np.diff(self.r_nodes)
        self.dr_nodes = np.append(self.dr_nodes, self.dr_nodes[-1])
        
        self.airfoil = Airfoil() # Single generic airfoil for V4.3.2

    def update(self, state, t, v_wind_hub, pitch_cmd_deg, unsteady=False):
        # Extract Scalars
        v_surge = state[22]
        v_pitch = state[26]
        rotor_speed = state[33]
        rotor_azimuth = state[11]
        
        # Extract Blade Flex Velocities (Flap1 for B1, B2, B3)
        q_dot_flaps = np.array([state[35], state[38], state[41]])
        
        pitch_cmd_rad = np.radians(pitch_cmd_deg)
        
        # Call JIT Kernel
        #F_gen, thrust, torque = calculate_blade_forces_jit(
            #3, self.p.R, self.p.HubHeight, self.rho,
            #self.r_nodes, self.dr_nodes, self.chord_nodes, self.twist_nodes,
            #self.airfoil.static_data,
           # v_surge, v_pitch, rotor_speed, rotor_azimuth,
           #q_dot_flaps, v_wind_hub, pitch_cmd_rad
        #)
        (
            F_gen,
            thrust,
            torque,
            phi,  # 新增：入流角分布
            alpha,  # 新增：攻角分布
            a,  # 新增：轴向诱导因子
            ap,  # 新增：切向诱导因子
            cn,  # 新增：法向力系数
            ct  # 新增：切向力系数
        ) = calculate_blade_forces_jit(3, self.p.R, self.p.HubHeight, self.rho,
            self.r_nodes, self.dr_nodes, self.chord_nodes, self.twist_nodes,
            self.airfoil.static_data,
            v_surge, v_pitch, rotor_speed, rotor_azimuth,
            q_dot_flaps, v_wind_hub, pitch_cmd_rad)

        self.last_thrust = thrust
        self.last_torque = torque
        #新添加部分
        self.phi = phi
        self.alpha = alpha
        self.a = a
        self.ap = ap
        self.cn = cn
        self.ct = ct
        
        return F_gen