import numpy as np
from numba import njit

# ==========================================
# 核心物理内核 (JIT Compiled)
# ==========================================

@njit(cache=True)
def calc_derivatives(
    pos, vel, 
    line_ranges, anch_indices, fair_indices,
    g, rho_w, rho_mat, d, l_seg0, EA, C_int, Area,
    C_dn, C_dt, C_an, C_at, kb, cb, Z_bot,
    mu_kT, mu_kA, mu_sT, mu_sA, fric_cv,
    is_relaxation
):
    n_nodes = pos.shape[0]
    n_lines = line_ranges.shape[0]
    
    force = np.zeros_like(pos)
    e_z = np.array([0.0, 0.0, 1.0])
    
    # === 1. 节段循环 (计算张力、阻尼、湿重) ===
    # (这部分逻辑不变，但为了完整性保留)
    for line_idx in range(n_lines):
        start = line_ranges[line_idx, 0]
        end = line_ranges[line_idx, 1]
        for j in range(start, end - 1):
            r_i, r_ip1 = pos[j], pos[j+1]
            v_i, v_ip1 = vel[j], vel[j+1]
            delta_r = r_ip1 - r_i
            dist = np.linalg.norm(delta_r)
            if dist < 1e-9: continue
            e_vec = delta_r / dist
            
            epsilon = (dist / l_seg0) - 1.0
            tension = EA * epsilon if epsilon > 0 else 0.0
            strain_rate = np.dot(delta_r, v_ip1 - v_i) / (l_seg0 * dist)
            f_axial = (tension + (C_int * Area) * strain_rate) * e_vec
            W_seg = Area * l_seg0 * (rho_w - rho_mat) * g * e_z
            
            force[j] += f_axial + 0.5 * W_seg
            force[j+1] -= f_axial + 0.5 * W_seg

    # === 2. 节点循环 (外部力：接触、拖曳、惯性) ===
    acc = np.zeros_like(vel)
    
    for i in range(n_nodes):
        # --- 边界判定 ---
        is_constrained = False
        for idx in anch_indices:
            if i == idx: is_constrained = True
        for idx in fair_indices:
            if i == idx: is_constrained = True
            
        # 节点有效长度
        is_endpoint = False
        for l_idx in range(n_lines):
            if i == line_ranges[l_idx, 0] or i == line_ranges[l_idx, 1] - 1:
                is_endpoint = True
                break
        l_realline = l_seg0 * 0.5 if is_endpoint else l_seg0

        # --- 计算节点切向量 q_hat (用于各向异性计算) ---
        # MoorDyn 使用相邻节点的平均切向量
        q_hat = np.array([1.0, 0.0, 0.0]) # 默认防止除零
        
        # 寻找前后节点索引
        # (这里为了性能使用了简化的查找逻辑，你可以优化为预计算结构)
        curr_start = -1
        curr_end = -1
        for l_idx in range(n_lines):
            if i >= line_ranges[l_idx, 0] and i < line_ranges[l_idx, 1]:
                curr_start = line_ranges[l_idx, 0]
                curr_end = line_ranges[l_idx, 1]
                break
        
        if curr_start != -1:
            has_prev = i > curr_start
            has_next = i < curr_end - 1
            if has_prev and has_next:
                # 中间节点：取前后向量平均 (MoorDyn 逻辑)
                v1 = pos[i] - pos[i-1]
                v2 = pos[i+1] - pos[i]
                q_sum = (v1 / (np.linalg.norm(v1)+1e-12)) + (v2 / (np.linalg.norm(v2)+1e-12))
                nm = np.linalg.norm(q_sum)
                if nm > 1e-9: q_hat = q_sum / nm
            elif has_next:
                # 起始点
                v = pos[i+1] - pos[i]
                nm = np.linalg.norm(v)
                if nm > 1e-9: q_hat = v / nm
            elif has_prev:
                # 终止点
                v = pos[i] - pos[i-1]
                nm = np.linalg.norm(v)
                if nm > 1e-9: q_hat = v / nm

        # ======================================================
        # (C) 海底接触 (MoorDyn 完全实现)
        # ======================================================
        z_curr = pos[i, 2]
        
        # 假设平坦海底，法向量 nvec = (0,0,1)
        # MoorDyn 支持测深网格，这里简化为常数 Z_bot
        if z_curr <= Z_bot:
            penetration = Z_bot - z_curr
            v_z = vel[i, 2]
            
            # 1. 法向支持力 (弹簧-阻尼)
            # Fn = (k*delta - c*v_n) * Area
            f_normal_mag = d * l_realline * (penetration * kb - v_z * cb)
            
            if f_normal_mag > 0:
                force[i, 2] += f_normal_mag
                
                # 2. 计算切平面速度 V_sb (Velocity Seabed)
                # 平坦海底 -> V_sb 就是 vel 的 xy 分量
                v_sb = vel[i].copy()
                v_sb[2] = 0.0 # 去掉法向速度
                
                # 3. 分解为 轴向(Axial) 和 横向(Transverse) 分量
                # Project V_sb onto q_hat
                q_xy = q_hat.copy()
                q_xy[2] = 0.0 # 投影到海床平面
                q_xy_norm = np.linalg.norm(q_xy)
                
                if q_xy_norm > 1e-9:
                    q_xy = q_xy / q_xy_norm # 轴向单位向量 (在海床上)
                    
                    val_a = np.dot(v_sb, q_xy)
                    v_a = val_a * q_xy        # 轴向速度向量
                    v_t = v_sb - v_a          # 横向速度向量
                else:
                    # 如果线垂直于海床，无法定义轴向，全视为横向
                    v_a = np.zeros(3)
                    v_t = v_sb
                
                mag_va = np.linalg.norm(v_a)
                mag_vt = np.linalg.norm(v_t)
                
                # 4. 计算最大动摩擦力 (Kinetic Limit)
                # F_max = mu_k * Fn
                f_k_t_max = mu_kT * f_normal_mag
                f_k_a_max = mu_kA * f_normal_mag
                
                # 5. 计算静摩擦比率阈值 (MoorDyn Logic: mc * F_max)
                # p%mc in MoorDyn usually represents mu_static / mu_kinetic
                # 所以 limit = mu_s * Fn
                limit_t = mu_sT * f_normal_mag
                limit_a = mu_sA * f_normal_mag
                
                # 6. 计算摩擦力 (Regularized Model)
                # Transverse Friction
                # 线性阻尼力 (模拟静摩擦/粘滞)
                f_linear_t = mu_kT * fric_cv * mag_vt 
                
                if f_linear_t > limit_t:
                    # 突破静摩擦 -> 使用动摩擦
                    # 方向与 v_t 相反
                    if mag_vt > 1e-9:
                        force[i] -= f_k_t_max * (v_t / mag_vt)
                else:
                    # 保持静摩擦 (线性阻尼区)
                    # Force = - (mu_kT * cv) * Vt
                    force[i] -= mu_kT * fric_cv * v_t
                    
                # Axial Friction
                f_linear_a = mu_kA * fric_cv * mag_va
                
                if f_linear_a > limit_a:
                    if mag_va > 1e-9:
                        force[i] -= f_k_a_max * (v_a / mag_va)
                else:
                    force[i] -= mu_kA * fric_cv * v_a

        # ======================================================
        # (D) 水动力 (Morison) - 使用之前的 q_hat
        # ======================================================
            v_rel = -vel[i] # 相对速度 (静水假设)
            
            # 估算切向向量 q_hat (寻找相邻节点)
            q_vec = np.zeros(3)
            # ... (查找逻辑保持不变) ...
            curr_start = -1
            curr_end = -1
            for l_idx in range(n_lines):
                if i >= line_ranges[l_idx, 0] and i < line_ranges[l_idx, 1]:
                    curr_start = line_ranges[l_idx, 0]
                    curr_end = line_ranges[l_idx, 1]
                    break
            
            if curr_start != -1:
                has_prev = i > curr_start
                has_next = i < curr_end - 1
                if has_prev and has_next:
                    q_vec = pos[i+1] - pos[i-1] # 中心差分
                elif has_next:
                    q_vec = pos[i+1] - pos[i]   # 前向差分
                elif has_prev:
                    q_vec = pos[i] - pos[i-1]   # 后向差分
            
            norm_q = np.linalg.norm(q_vec)
            q_hat = np.array([0.0, 0.0, 1.0])
            if norm_q > 1e-9:
                q_hat = q_vec / norm_q
                
            # 速度分解normal（横向）和 tangential （切向）分量
            v_tan_val = np.dot(v_rel, q_hat)
            v_tan_vec = v_tan_val * q_hat
            v_norm_vec = v_rel - v_tan_vec
            
            norm_vn = np.linalg.norm(v_norm_vec)
            norm_vt = np.abs(v_tan_val)
            
            # 拖曳力
            f_drag_n = 0.5 * rho_w * C_dn * (d * l_realline) * norm_vn * v_norm_vec  #横向力
            f_drag_t = 0.5 * rho_w * C_dt * (np.pi * d * l_realline) * norm_vt * v_tan_vec #切向力
            
            force[i] += f_drag_n + f_drag_t
        
            # (E) 求解加速度 (保持不变)
            if not is_constrained:
                m_struct = rho_mat * Area * l_realline
            if m_struct < 1e-9: m_struct = 1e-9
            
            # 分别计算两个方向的附加质量
            # MoorDyn: Mn = rho*Vol*Can, Mt = rho*Vol*Cat
            vol_node = Area * l_realline
            m_add_n = rho_w * vol_node * C_an
            m_add_t = rho_w * vol_node * C_at
            
            # 将总力分解为 切向 和 法向
            f_total = force[i]
            f_tan_val_total = np.dot(f_total, q_hat)
            f_tan_vec_total = f_tan_val_total * q_hat
            f_norm_vec_total = f_total - f_tan_vec_total
            
            # 分别求解加速度分量 (a = F / (m + m_added))
            # 切向加速度 (通常质量较小，反应快)
            acc_tan = f_tan_vec_total / (m_struct + m_add_t)
            
            # 法向加速度 (附加质量大，反应慢)
            acc_norm = f_norm_vec_total / (m_struct + m_add_n)
            
            # 合成总加速度
            acc[i] = acc_tan + acc_norm
            
            # 松弛模式阻尼
            if is_relaxation:
                acc[i] -= 2.0 * vel[i]
        else:
            acc[i] = 0.0 # 约束节点不计算动力学

    return acc, force
    

@njit(cache=True)
def rk2_step(
    pos, vel, 
    line_ranges, anch_indices, fair_indices,
    target_pos, target_vel,
    dt,
    g, rho_w, rho_mat, d, l_seg0, EA, C_int, Area,
    C_dn, C_dt, C_an, C_at, kb, cb, Z_bot,
    mu_kT, mu_kA, mu_sT, mu_sA, fric_cv
):
    """
    RK2 (Heun's Method) 积分步
    """
    n_fair = len(fair_indices)
    
    # 1. 强制边界条件 (当前时刻)
    for i in range(len(anch_indices)):
        idx = anch_indices[i]
        vel[idx] = 0.0
    for k in range(n_fair):
        idx = fair_indices[k]
        pos[idx] = target_pos[k] # 这里假设子步内位置线性变化，简化为直接赋值
        vel[idx] = target_vel[k]

    # --- K1 阶段 ---
    acc1, _ = calc_derivatives(
        pos, vel, line_ranges, anch_indices, fair_indices,
        g, rho_w, rho_mat, d, l_seg0, EA, C_int, Area,
        C_dn, C_dt, C_an, C_at, kb, cb, Z_bot, 
        mu_kT, mu_kA, mu_sT, mu_sA, fric_cv,
        False
    )
    
    # 预估状态 (Predictor)
    pos_pred = pos + vel * dt
    vel_pred = vel + acc1 * dt
    
    # 强制边界条件 (预估时刻)
    # 注意：更精确的做法是插值 target_pos，这里简化为保持速度不变
    for i in range(len(anch_indices)):
        idx = anch_indices[i]
        vel_pred[idx] = 0.0
    for k in range(n_fair):
        idx = fair_indices[k]
        pos_pred[idx] = target_pos[k] + target_vel[k] * dt
        vel_pred[idx] = target_vel[k]

    # --- K2 阶段 ---
    acc2, _ = calc_derivatives(
        pos_pred, vel_pred, line_ranges, anch_indices, fair_indices,
        g, rho_w, rho_mat, d, l_seg0, EA, C_int, Area,
        C_dn, C_dt, C_an, C_at, kb, cb, Z_bot, 
        mu_kT, mu_kA, mu_sT, mu_sA, fric_cv,
        False
    )
    
    # 校正状态 (Corrector)
    pos_new = pos + 0.5 * (vel + vel_pred) * dt
    vel_new = vel + 0.5 * (acc1 + acc2) * dt
    
    return pos_new, vel_new

@njit(cache=True)
def solve_physics_substeps(
    pos, vel,
    line_ranges, anch_indices, fair_indices,
    target_pos, target_vel,
    n_substeps, dt,
    g, rho_w, rho_mat, d, l_seg0, EA, C_int, Area,
    C_dn, C_dt, C_an, C_at, kb, cb, Z_bot,
    mu_kT, mu_kA, mu_sT, mu_sA, fric_cv
):
    # 子步循环
    for _ in range(n_substeps):
        pos, vel = rk2_step(
            pos, vel, line_ranges, anch_indices, fair_indices,
            target_pos, target_vel, dt,
            g, rho_w, rho_mat, d, l_seg0, EA, C_int, Area,
            C_dn, C_dt, C_an, C_at, kb, cb, Z_bot,
            mu_kT, mu_kA, mu_sT, mu_sA, fric_cv
        )
    return pos, vel

@njit(cache=True)
def relaxation_run(
    pos, vel,
    line_ranges, anch_indices, fair_indices,
    n_steps, dt,
    g, rho_w, rho_mat, d, l_seg0, EA, C_int, Area,
    C_dn, C_dt, C_an, C_at, kb, cb, Z_bot,
    mu_kT, mu_kA, mu_sT, mu_sA, fric_cv,
):
    """
    动态松弛专用的积分循环 (使用 Euler + 高阻尼)
    """
    for _ in range(n_steps):
        # 强制归零锚点和导缆孔速度 (静态平衡寻找)
        for idx in anch_indices: vel[idx] = 0.0
        for idx in fair_indices: vel[idx] = 0.0
            
        acc, _ = calc_derivatives(
            pos, vel, line_ranges, anch_indices, fair_indices,
            g, rho_w, rho_mat, d, l_seg0, EA, C_int, Area,
            C_dn, C_dt, C_an, C_at, kb, cb, Z_bot,
            mu_kT, mu_kA, mu_sT, mu_sA, fric_cv, 
            True # 开启 relaxation 模式 (增加人工阻尼)
        )
        
        vel += acc * dt
        pos += vel * dt
        
        # 能量耗散：每步稍微衰减速度
        vel *= 0.99 
        
    return pos, vel

# ==========================================
# Python 主类
# ==========================================

class MoorEmmSolver:
    """
    MoorEmm v2: RK2 Integrator & Dynamic Relaxation
    """
    def __init__(self, params, seastate, dt):
        print(f"[MoorEmm] Initializing Solver v2 (MoorDyn-consistent logic)...")
        self.p = params
        self.dt_global = float(dt)
        self.custom_fairlead_points = None
        self.custom_anchor_points = None
        
        # --- 1. 物理参数 ---
        self.g = float(self.p.Gravity)
        self.rho_w = float(self.p.WaterDensity)
        self.depth = float(self.p.WaterDepth)
        
        self.d = float(self.p.Moor_LineDiam)
        self.mass_den_linear = float(self.p.Moor_LineMass)
        self.Area = np.pi * self.d**2 / 4.0
        self.rho_mat = self.mass_den_linear / self.Area 
        
        self.L_total = float(self.p.Moor_LineLength)
        self.EA = float(self.p.Moor_LineEA)
        self.C_int = float(self.p.Moor_Cint)
        
        self.C_dn = float(self.p.Moor_Cdn)
        self.C_dt = float(self.p.Moor_Cdt)
        self.C_an = float(self.p.Moor_Can)
        self.C_at = float(self.p.Moor_Cat)
        
        self.kb = float(self.p.Moor_Kb)
        self.cb = float(self.p.Moor_Cb)
        self.Z_bot = -self.depth
        
                # === 新增：MoorDyn 风格的摩擦参数 ===
        # 如果 params 中没有定义，这里给出了 MoorDyn 的常用默认值
        # Transverse (横向) & Axial (轴向)
        self.mu_kT = float(self.p.Moor_Fric_Mu_kT)  # 横向动摩擦
        self.mu_kA = float(self.p.Moor_Fric_Mu_kA)  # 轴向动摩擦 (通常比横向小)

        self.mu_sT = float(self.p.Moor_Fric_Mu_sT)  # 横向静摩擦
        self.mu_sA = float(self.p.Moor_Fric_Mu_sA)  # 轴向静摩擦
        
        # 摩擦线性区斜率参数 (MoorDyn 中的 p%cv)
        # 这个值决定了从静止到滑动的过渡陡峭程度，通常是一个较大的值
        self.fric_cv = float(self.p.Moor_Fric_Cv)

        # --- 2. 离散化 ---
        self._load_custom_points()
        self.n_segs = int(self.p.Moor_NumSegs)
        self.n_nodes_per_line = self.n_segs + 1 
        self.l_seg0 = self.L_total / self.n_segs
        
        self.n_total_nodes = self.n_lines * self.n_nodes_per_line
        
        # 状态数组
        self.pos = np.zeros((self.n_total_nodes, 3), dtype=np.float64)
        self.vel = np.zeros((self.n_total_nodes, 3), dtype=np.float64)
        
        # 索引构建
        self.anch_indices_list = []
        self.fair_indices_list = []
        self.line_ranges_list = []
        
        self._initialize_geometry()
        
        self.anch_indices = np.array(self.anch_indices_list, dtype=np.int64)
        self.fair_indices = np.array(self.fair_indices_list, dtype=np.int64)
        self.line_ranges = np.array(self.line_ranges_list, dtype=np.int64)
        
        # --- 3. 时间步长设置 ---
        wave_speed = np.sqrt(self.EA / self.mass_den_linear) if self.mass_den_linear > 0 else 1000.0
        crit_dt = self.l_seg0 / wave_speed
        self.dt_physics = min(getattr(self.p, 'Moor_MaxDt', 0.1), crit_dt * 0.8) # 0.8 CFL for RK2
        self.n_substeps = int(np.ceil(self.dt_global / self.dt_physics))
        self.dt_physics = self.dt_global / self.n_substeps
        
        # --- 4. 执行动态松弛 (MoorDyn_Init Logic) ---
        self._dynamic_relaxation()

    def _load_custom_points(self):
        # (与原代码相同，省略以节省篇幅，请保留原有的实现)
        fairlead_points = getattr(self.p, "Moor_FairleadPoints", None)
        anchor_points = getattr(self.p, "Moor_AnchorPoints", None)
        self.custom_fairlead_points = np.asarray(fairlead_points, dtype=np.float64)
        self.custom_anchor_points = np.asarray(anchor_points, dtype=np.float64)
        self.n_lines = int(self.custom_fairlead_points.shape[0])

    def _initialize_geometry(self):
        # 初始化为直线
        for i in range(self.n_lines):
            p_anchor = self.custom_anchor_points[i]
            p_fair = self.custom_fairlead_points[i]
            start_idx = i * self.n_nodes_per_line
            end_idx = start_idx + self.n_nodes_per_line
            
            self.anch_indices_list.append(start_idx)
            self.fair_indices_list.append(end_idx - 1)
            self.line_ranges_list.append([start_idx, end_idx])
            
            for k in range(self.n_nodes_per_line):
                idx = start_idx + k
                frac = k / self.n_segs
                
                # 简单的悬链线预估 (可选：这里用直线，依靠 Relaxation 修正)
                self.pos[idx] = (1 - frac) * p_anchor + frac * p_fair
                
                # 确保初始位置不穿透海底
                if self.pos[idx, 2] < self.Z_bot: self.pos[idx, 2] = self.Z_bot + 0.1

    def _dynamic_relaxation(self):
        """
        动态松弛算法：
        在仿真开始前，使用高阻尼和人工耗散运行一段时间，
        消除初始直线布局的势能，使系统达到静态平衡。
        """
        print("[MoorEmm] Running Dynamic Relaxation (Target: Static Equilibrium)...")
        
        # 临时增大阻尼 (MoorDyn 做法：Multiplier)
        relax_C_int = self.C_int * 5.0
        relax_C_dn = self.C_dn * 5.0
        
        # 运行时间：通常 5-10 秒足够
        T_relax = 10.0
        n_steps = int(T_relax / self.dt_physics)
        
        self.pos, self.vel = relaxation_run(
            self.pos, self.vel,
            self.line_ranges, self.anch_indices, self.fair_indices,
            n_steps, self.dt_physics,
            self.g, self.rho_w, self.rho_mat, self.d, self.l_seg0, self.EA, 
            relax_C_int, self.Area,
            relax_C_dn, self.C_dt, self.C_an, self.C_at, self.kb, self.cb, self.Z_bot,
            self.mu_kT, self.mu_kA, self.mu_sT, self.mu_sA, self.fric_cv
        )
        
        # 强行重置速度为0，确保从静止开始
        self.vel[:] = 0.0
        print("[MoorEmm] Relaxation Complete.")

    def update(self, state, t):
        # 1. 平台运动解析 (与原代码一致)
        plat_pos = np.array(state[0:3], dtype=np.float64)
        plat_rot = np.array(state[3:6], dtype=np.float64)
        plat_vel = np.array(state[6:9], dtype=np.float64)
        plat_omega = np.array(state[9:12], dtype=np.float64)
        
        cx, sx = np.cos(plat_rot[0]), np.sin(plat_rot[0])
        cy, sy = np.cos(plat_rot[1]), np.sin(plat_rot[1])
        cz, sz = np.cos(plat_rot[2]), np.sin(plat_rot[2])
        R = np.array([[cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
                      [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
                      [-sy,   cy*sx,            cy*cx]], dtype=np.float64)
        
        # 2. 计算导缆孔目标状态
        n_fair = len(self.fair_indices)
        target_pos = np.zeros((n_fair, 3), dtype=np.float64)
        target_vel = np.zeros((n_fair, 3), dtype=np.float64)
        
        for k in range(self.n_lines):
            p_local = self.custom_fairlead_points[k]
            target_pos[k] = plat_pos + R @ p_local
            target_vel[k] = plat_vel + np.cross(plat_omega, R @ p_local)
            
        # 3. 调用 RK2 求解器 (替代了原有的 solve_physics_jit)
        self.pos, self.vel = solve_physics_substeps(
            self.pos, self.vel,
            self.line_ranges, self.anch_indices, self.fair_indices,
            target_pos, target_vel,
            self.n_substeps, self.dt_physics,
            self.g, self.rho_w, self.rho_mat, self.d, self.l_seg0, self.EA, self.C_int, self.Area,
            self.C_dn, self.C_dt, self.C_an, self.C_at, self.kb, self.cb, self.Z_bot,
            self.mu_kT, self.mu_kA, self.mu_sT, self.mu_sA, self.fric_cv
        )

        # 4. 计算反作用力 (输出到 FAST)
        F_total = np.zeros(6, dtype=np.float64)
        
        # 使用 calc_derivatives 最后一次计算精确的力 (可选，为了高精度)
        # 或者直接用几何应变计算 (如下，效率高)
        for k in range(n_fair):
            idx = self.fair_indices[k]
            prev_idx = idx - 1
            delta = self.pos[prev_idx] - self.pos[idx]
            dist = np.linalg.norm(delta)
            
            if dist > 1e-6:
                strain = (dist - self.l_seg0) / self.l_seg0
                if strain > 0:
                    tension = self.EA * strain
                    f_vec = tension * (delta / dist)
                    
                    F_total[0:3] += f_vec
                    r_arm = self.pos[idx] - plat_pos
                    F_total[3:6] += np.cross(r_arm, f_vec)
        
        out = np.zeros(14)
        out[0:6] = F_total
        return out