import numpy as np
from numba import njit

# --- Numba Kernels (High Performance Calculation) ---

@njit(cache=True)
def calc_irregular_wave_kinematics_jit(t, x, y, z_arr, wave_comps):              #这是一个 NumPy + Numba 的核函数，
    """                                                                          #用来在同一个水平位置 (x, y) 、多个垂向位置 z_arr 上
    批量计算多个 Z 高度处的波浪运动学 (Superposition)                            #计算不规则波的速度和加速度
    wave_comps: [N_freq, 4] array (w, k, amp, phi)
    z_arr: [N_nodes] array
    Returns: vel_w [N_nodes, 3], acc_w [N_nodes, 3]
    """
    n_nodes = len(z_arr)
    n_freq = len(wave_comps)
    
    vel_w = np.zeros((n_nodes, 3))
    acc_w = np.zeros((n_nodes, 3))
    
    for i in range(n_nodes):
        z = z_arr[i]
        # Wheeler stretching / cut-off
        z_eff = z if z < 0 else 0.0
        
        u_sum = 0.0
        w_sum = 0.0
        ax_sum = 0.0
        az_sum = 0.0
        
        for j in range(n_freq):                  #对每个频率分量做 Airy 叠加
            w = wave_comps[j, 0]
            k = wave_comps[j, 1]
            A = wave_comps[j, 2]
            phi = wave_comps[j, 3]
            
            phase = k*x - w*t + phi
            decay = np.exp(k * z_eff)
            
            s_ph = np.sin(phase)
            c_ph = np.cos(phase)
            
            u_sum += A * w * decay * c_ph
            w_sum += A * w * decay * s_ph
            ax_sum += A * w**2 * decay * s_ph
            az_sum -= A * w**2 * decay * c_ph
            
        vel_w[i, 0] = u_sum
        vel_w[i, 2] = w_sum
        acc_w[i, 0] = ax_sum                       #把每个点的水平/竖向速度和加速度放进数组，返回速度和加速度
        acc_w[i, 2] = az_sum
        
    return vel_w, acc_w

@njit(cache=True)
def calc_strip_forces_jit(
    t, surge, sway, heave, roll, pitch, yaw,
    v_surge, v_sway, v_heave, v_roll, v_pitch, v_yaw,
    mem_x, mem_y, mem_z_nodes, mem_dz, mem_diam, mem_cd, mem_ca, 
    wave_comps, rho, wave_mode, reg_H, reg_T, g
):
    """
    Morison Integration Kernel
    """
    F_sum = np.zeros(6)
    n_strips = len(mem_z_nodes)
    
    for i in range(n_strips):                   # 遍历每个条带
        mx = mem_x[i]
        my = mem_y[i]
        mz = mem_z_nodes[i]
        
        # 1. Global Position update     线性化的刚体位移变换，把条带中心从局部坐标变到全局 z 坐标
        z_g = mz + heave + pitch*mx - roll*my
        if z_g > 0: continue
        
        # 2. Struct Velocity                    条带的结构速度
        v_sx = v_surge + v_pitch*mz - v_yaw*my
        v_sy = v_sway - v_roll*mz   + v_yaw*mx
        v_sz = v_heave + v_roll*my  - v_pitch*mx
        
        # 3. Wave Kinematics
        u_w = 0.0; w_w = 0.0; ax_w = 0.0; az_w = 0.0
        
        if wave_mode == 1: # Regular   规则波
            if reg_H > 0:
                w = 2*np.pi/(reg_T + 1e-6)
                k = w**2/g
                decay = np.exp(k * z_g)
                ph = k*mx - w*t # Assume wave along X
                u_w = 0.5 * reg_H * w * decay * np.cos(ph)
                w_w = 0.5 * reg_H * w * decay * np.sin(ph)
                ax_w = 0.5 * reg_H * w**2 * decay * np.sin(ph)
                az_w = -0.5 * reg_H * w**2 * decay * np.cos(ph)
        
        elif wave_mode >= 2: # Irregular  不规则波
             # Inline calculation for speed
             # Numba 知道 wave_comps 是 2D 数组，即使长度为 0，类型也是 float64[:,:]
             for j in range(len(wave_comps)):
                wc = wave_comps[j]
                wk = wc[1]
                decay = np.exp(wk * z_g)
                ph = wk*mx - wc[0]*t + wc[3]
                s_ph = np.sin(ph); c_ph = np.cos(ph)
                
                u_w += wc[2] * wc[0] * decay * c_ph
                w_w += wc[2] * wc[0] * decay * s_ph
                ax_w += wc[2] * wc[0]**2 * decay * s_ph
                az_w -= wc[2] * wc[0]**2 * decay * c_ph

        # 4. Morison Force
        area = np.pi * (mem_diam[i]/2.0)**2   #条带截面积
        
        # Inertia              #惯性项
        fx_in = rho * area * mem_dz[i] * (1.0 + mem_ca[i]) * ax_w
        fz_in = rho * area * mem_dz[i] * (1.0 + mem_ca[i]) * az_w
        
        # Drag                  #阻力项
        vx_rel = u_w - v_sx
        vy_rel = 0.0 - v_sy
        vz_rel = w_w - v_sz
        v_rel_mag = np.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2)
        
        const_dr = 0.5 * rho * mem_diam[i] * mem_dz[i] * mem_cd[i]
        fx_dr = const_dr * vx_rel * v_rel_mag
        fy_dr = const_dr * vy_rel * v_rel_mag
        fz_dr = const_dr * vz_rel * v_rel_mag
        
        fx = fx_in + fx_dr
        fy = fy_dr
        fz = fz_in + fz_dr
        
        # Accumulate                       # 积分到 6DOF：力与力矩累加
        F_sum[0] += fx
        F_sum[1] += fy
        F_sum[2] += fz
        F_sum[3] += my*fz - mz*fy
        F_sum[4] += mz*fx - mx*fz
        F_sum[5] += mx*fy - my*fx
        
    return F_sum                           #返回 6 维广义力向量（不含静水和辐射阻尼）

# --- Main Class ---

class HydroDynamics:
    """
    WOUSE V4.3.2 Hydro Module (Fixed Numba Typing)
    """
    def __init__(self, params, seastate):
        self.p = params
        self.seastate = seastate
        self.rho = self.p.WaterDensity
        self.g = self.p.Gravity
        
        # Flatten Geometry for JIT          #创建一堆空列表，存储条带几何：中心坐标、长度、直径、阻力系数、附加质量系数，根据 OC4 几何填充这些列表
        self.arr_x = []
        self.arr_y = []
        self.arr_z = []
        self.arr_dz = []
        self.arr_D = []
        self.arr_Cd = []
        self.arr_Ca = []
        
        self._init_geometry_arrays()
        
        # Convert to numpy for Numba        #把列表转为 numpy 数组，以便传进 Numba JIT 核函数
        self.arr_x = np.array(self.arr_x)
        self.arr_y = np.array(self.arr_y)
        self.arr_z = np.array(self.arr_z)
        self.arr_dz = np.array(self.arr_dz)
        self.arr_D = np.array(self.arr_D)
        self.arr_Cd = np.array(self.arr_Cd)
        self.arr_Ca = np.array(self.arr_Ca)
        
        # PotFlow Matrices (Simplified for V4.3)
        self.B_rad = np.zeros((6,6)); self.B_rad[2,2] = 1.5e6; self.B_rad[4,4] = 5.0e8           #6×6 辐射阻尼矩阵
        self.C_hydro = np.zeros((6,6)); self.C_hydro[2,2] = 3.8e6; self.C_hydro[4,4] = 1.0e9      #6×6 线性静水刚度矩阵
        
        # Static Equilibrium         #静力平衡（浮力－重量）
        self.net_static = self._calc_static_buoyancy() - (1.34e7 + 3.5e5 + 3.5e5)*9.81

    def _init_geometry_arrays(self):         #几何的初始化
        # OC4 Geometry
        r = 28.87
        m_defs = [("Main", 0,0, -20, 10, 6.5)]
        for deg in [0, 120, 240]:
            rad = np.radians(deg)
            x, y = r*np.cos(rad), r*np.sin(rad)
            m_defs.append((f"Up_{deg}", x, y, -14, 12, 12.0))
            m_defs.append((f"Base_{deg}", x, y, -20, -14, 24.0))
            
        for m in m_defs:
            n_strips = 10
            dz = (m[4] - m[3]) / n_strips
            z_start = m[3] + dz/2
            for i in range(n_strips):
                self.arr_x.append(m[1])
                self.arr_y.append(m[2])
                self.arr_z.append(z_start + i*dz)
                self.arr_dz.append(dz)
                self.arr_D.append(m[5])
                self.arr_Cd.append(0.6)
                self.arr_Ca.append(0.97)

    def _calc_static_buoyancy(self):          #净水浮力计算
        vol = np.sum(self.arr_dz[self.arr_z < 0] * (np.pi * (self.arr_D[self.arr_z < 0]/2)**2))
        return vol * self.rho * self.g

    def update(self, state, t):
        """
        state: 12-DOF rigid body state [x,y,z,r,p,y, vx,vy,vz,vr,vp,vy]
        """
        # Unpack State
        surge, sway, heave, roll, pitch, yaw = state[:6]
        
        # [V4.3.2 Fix] Correct index for 12-DOF rigid_state
        v_surge, v_sway, v_heave, v_roll, v_pitch, v_yaw = state[6:12]
        
        # [V4.3.2 Fix] Ensure wave_comps_arr is always 2D for Numba
        # Shape must be (N, 4) even if N=0
        wave_comps_arr = np.zeros((0, 4), dtype=np.float64) 
        
        if self.p.Env_WaveMod in [2, 3] and hasattr(self.seastate, 'wave_components'):
            comps = self.seastate.wave_components
            if len(comps) > 0:
                wave_comps_arr = np.array(comps, dtype=np.float64)
            
        # Call JIT Kernel       #调用前面解释过的 calc_strip_forces_jit ，得到 Morison 条带力积分的结果 F_strip
        F_strip = calc_strip_forces_jit(
            t, surge, sway, heave, roll, pitch, yaw,
            v_surge, v_sway, v_heave, v_roll, v_pitch, v_yaw,
            self.arr_x, self.arr_y, self.arr_z, self.arr_dz, self.arr_D, self.arr_Cd, self.arr_Ca,
            wave_comps_arr, self.rho, 
            self.p.Env_WaveMod, self.p.Env_WaveHeight, self.p.Env_WavePeriod, self.g
        )
        
        # Add Hydrostatics
        F_static = np.zeros(6)
        F_static[2] = self.net_static       #静力平衡得到的
        F_static -= self.C_hydro @ state[:6]    # 线性静水复原力
        
        F_total = np.zeros(14)
        F_total[:6] = F_strip + F_static         # 总水动力：条带力 + 静水力
        
        # Hybrid Damping
        if self.p.Hydro_Method >= 1:          #若不用莫里森条带力，则只用线性辐射阻尼
            F_total[:6] -= self.B_rad @ state[6:12]      #辐射阻尼力
            
        return F_total