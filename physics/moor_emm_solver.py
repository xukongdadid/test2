import numpy as np

class MoorEmmSolver:
    """
    MoorEmm: Python Native Lumped-Mass Mooring Solver
    文件名: moor_emm_solver.py
    
    
    主要特性:
    - [cite_start]离散化: N段, N+1节点 [cite: 223]
    - 受力: 轴向张力(EA), 内部阻尼(C_int), 湿重(W), Morison水动力(切向/法向), 海底接触(弹簧阻尼)
    - [cite_start]动力学方程: [m+a]r_ddot = F_total [cite: 288]
    """
    def __init__(self, params, seastate, dt):
        print(f"[MoorEmm] Initializing Python Lumped-Mass Solver (Paper Implementation)...")
        self.p = params
        self.seastate = seastate
        self.dt_global = float(dt)
        
        # [cite_start]--- 1. 物理参数解析 (基于论文表3.5 [cite: 370, 371]) ---
        self.g = self.p.Gravity
        self.rho_w = self.p.WaterDensity # 1025 kg/m^3
        self.depth = self.p.WaterDepth
        
        # 缆索属性 (优先使用 params, 缺失则使用论文表3.5默认值)
        self.L_total = self.p.Moor_LineLength
        self.mass_den_linear = self.p.Moor_LineMass   # kg/m (线性密度)
        self.EA = self.p.Moor_LineEA                  # N (轴向刚度)
        
        # [cite_start]论文特定参数 [cite: 371]
        self.d = getattr(self.p, 'Moor_LineDiam', 0.0766) # 直径 d
        self.C_int = getattr(self.p, 'Moor_Cint', 1.221E7) # 内部阻尼系数 N-s
        
        # [cite_start]水动力系数 [cite: 371]
        self.C_dn = getattr(self.p, 'Moor_Cdn', 2.0)   # 横向拖曳系数
        self.C_dt = getattr(self.p, 'Moor_Cdt', 0.4)   # 切向拖曳系数
        self.C_an = getattr(self.p, 'Moor_Can', 0.8)   # 横向附加质量系数
        self.C_at = getattr(self.p, 'Moor_Cat', 0.25)  # 切向附加质量系数
        
        # [cite_start]海底接触参数 [cite: 371]
        self.kb = getattr(self.p, 'Moor_Kb', 3.0E6)    # 海床垂直刚度 Pa/m
        self.cb = getattr(self.p, 'Moor_Cb', 3.0E5)    # 海床垂直阻尼 Pa s/m
        self.Z_bot = -self.depth                       # 海床Z坐标
        
        # [cite_start]计算截面积 A = pi * d^2 / 4 [cite: 258]
        self.A_sect = np.pi * self.d**2 / 4.0
        
        # --- 2. 离散化设置 ---
        self.n_lines = int(self.p.Moor_NumLines)
        self.n_segs = 10  # N段 [cite: 223]
        self.n_nodes_per_line = self.n_segs + 1 # N+1 节点
        self.l_seg0 = self.L_total / self.n_segs # 初始原长 l [cite: 227]
        
        # 节点总数
        self.n_total_nodes = self.n_lines * self.n_nodes_per_line
        
        # --- 3. 初始化状态数组 ---
        # Pos: (N_total, 3), Vel: (N_total, 3)
        self.pos = np.zeros((self.n_total_nodes, 3))
        self.vel = np.zeros((self.n_total_nodes, 3))
        # 力数组
        self.force = np.zeros((self.n_total_nodes, 3))
        # [cite_start]节点结构质量 (标量) m_i = rho * A * l = mass_den_linear * l [cite: 285]
        # 注意: 论文中 m_i 为对角矩阵元素, 这里先存标量
        self.node_mass = np.zeros(self.n_total_nodes)
        
        # 索引管理
        self.anch_indices = [] # 锚点索引 (Fixed)
        self.fair_indices = [] # 导缆孔索引 (Kinematic Coupling)
        self.line_ranges = []  # 每根缆索的节点范围 (start, end)
        
        self._initialize_geometry_and_mass()
        
        # --- 4. 求解器稳定性设置 (Sub-stepping) ---
        # 显式积分 Courant 条件估算
        wave_speed = np.sqrt(self.EA / self.mass_den_linear)
        crit_dt = self.l_seg0 / wave_speed
        
        # 安全系数
        self.dt_physics = min(0.00125, crit_dt * 0.5) # 论文提及动态系泊步长0.00125s [cite: 375]
        self.n_substeps = int(np.ceil(self.dt_global / self.dt_physics))
        self.dt_physics = self.dt_global / self.n_substeps
        
        print(f"[MoorEmm] Setup: {self.n_lines} lines, {self.n_segs} segs/line.")
        print(f"[MoorEmm] Physics: Euler Integration with Sub-stepping: {self.n_substeps} steps (dt={self.dt_physics:.1e}s)")

    def _initialize_geometry_and_mass(self):
        """初始化节点位置与质量"""
        r_f = self.p.Moor_FairleadRadius
        z_f = -self.p.Moor_FairleadDraft
        r_a = self.p.Moor_AnchorRadius
        z_a = -self.p.Moor_AnchorDepth
        
        # [cite_start]节点结构质量 m = rho_linear * l [cite: 285]
        seg_mass = self.l_seg0 * self.mass_den_linear
        
        for i in range(self.n_lines):
            angle = i * (2*np.pi/self.n_lines) + np.pi
            
            p_anchor = np.array([r_a*np.cos(angle), r_a*np.sin(angle), z_a])
            p_fair = np.array([r_f*np.cos(angle), r_f*np.sin(angle), z_f])
            
            start_idx = i * self.n_nodes_per_line
            end_idx = start_idx + self.n_nodes_per_line
            
            self.anch_indices.append(start_idx)
            self.fair_indices.append(end_idx - 1)
            self.line_ranges.append((start_idx, end_idx))
            
            # 线性插值初始化位置
            for k in range(self.n_nodes_per_line):
                idx = start_idx + k
                frac = k / self.n_segs
                self.pos[idx] = (1 - frac) * p_anchor + frac * p_fair
                
                # 质量分配: 内部节点 = 1 seg mass; 端点 = 0.5 seg mass (近似处理)
                factor = 1.0
                if k == 0 or k == self.n_segs: factor = 0.5
                self.node_mass[idx] = seg_mass * factor

    def update(self, state, t):
        """
        Input: state [surge, sway, heave, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        Output: F_mooring (14-array)
        """
        # 1. 平台运动解析 (与原逻辑一致)
        plat_pos = np.array(state[0:3])
        plat_rot = np.array(state[3:6])
        plat_vel = np.array(state[6:9])
        plat_omega = np.array(state[9:12])
        
        cx, sx = np.cos(plat_rot[0]), np.sin(plat_rot[0])
        cy, sy = np.cos(plat_rot[1]), np.sin(plat_rot[1])
        cz, sz = np.cos(plat_rot[2]), np.sin(plat_rot[2])
        
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        R = Rz @ Ry @ Rx
        
        # 2. 更新导缆孔目标位置
        fair_targets = []
        fair_vels = []
        
        r_f0 = self.p.Moor_FairleadRadius
        z_f0 = -self.p.Moor_FairleadDraft
        
        for k in range(self.n_lines):
            angle = k * (2*np.pi/self.n_lines) + np.pi
            p_local = np.array([r_f0*np.cos(angle), r_f0*np.sin(angle), z_f0])
            
            p_global = plat_pos + R @ p_local
            v_global = plat_vel + np.cross(plat_omega, R @ p_local)
            
            fair_targets.append(p_global)
            fair_vels.append(v_global)
            
        # 3. 物理子步循环
        for _ in range(self.n_substeps):
            # 强制设置边界条件
            for k, idx in enumerate(self.fair_indices):
                self.pos[idx] = fair_targets[k]
                self.vel[idx] = fair_vels[k]
            
            for idx in self.anch_indices:
                self.vel[idx] = 0.0 # 锚点固定
            
            # 计算力 & 积分
            self._step_physics()

        # 4. 计算对平台的作用力 (Fairlead Tension)
        F_total = np.zeros(6)
        
        for k, idx in enumerate(self.fair_indices):
            # 取连接导缆孔的最后一节段 (节点 N-1 到 N)
            prev_idx = idx - 1
            delta = self.pos[prev_idx] - self.pos[idx] # 指向缆索内部
            L_curr = np.linalg.norm(delta)
            
            # 计算导缆孔处的张力 (类似于公式 3.47)
            if L_curr > 1e-6:
                strain = (L_curr - self.l_seg0) / self.l_seg0
                if strain < 0: strain = 0
                tension = self.EA * strain
                f_vec = tension * (delta / L_curr)
                
                F_total[0:3] += f_vec
                r_arm = self.pos[idx] - plat_pos
                F_total[3:6] += np.cross(r_arm, f_vec)
        
        out = np.zeros(14)
        out[0:6] = F_total
        return out

    def _step_physics(self):
        """
        物理计算单步: 严格遵循论文公式计算力与加速度
        Forces -> Acceleration -> Velocity -> Position
        """
        self.force.fill(0.0)
        
        # --- A. 内部作用力 (张力与阻尼) ---
        # 遍历每根缆索的每个节段 (节点 i 到 i+1)
        for start, end in self.line_ranges:
            for i in range(start, end - 1):
                # 节点 i 和 i+1
                r_i = self.pos[i]
                r_ip1 = self.pos[i+1]
                v_i = self.vel[i]
                v_ip1 = self.vel[i+1]
                
                delta_r = r_ip1 - r_i
                dist = np.linalg.norm(delta_r)
                
                if dist < 1e-8: continue
                
                dir_vec = delta_r / dist # 这里的方向是从 i 指向 i+1
                
                # 1. 计算轴向张力 T_{i+1/2}
                # T = EA * (|r_{i+1} - r_i| - l) / l
                strain = (dist - self.l_seg0) / self.l_seg0
                
                if strain > 0:
                    tension = self.EA * strain
                else:
                    tension = 0.0 # 松弛状态 T=0 [cite: 270]
                
                # 2. 计算内部阻尼力 C_{i+1/2}
                # C = C_int * strain_rate
                # strain_rate = (1/l) * ( (r_{i+1}-r_i) dot (v_{i+1}-v_i) ) / |r_{i+1}-r_i|
                delta_v = v_ip1 - v_i
                strain_rate = np.dot(delta_r, delta_v) / (self.l_seg0 * dist)
                
                damping_force_mag = 0.0
                if strain > 0: # 仅在拉伸时应用阻尼? 论文未明确，通常随刚度存在
                     damping_force_mag = self.C_int * strain_rate
                
                # 总轴向力标量 (拉力为正)
                total_axial = tension + damping_force_mag
                
                # 力向量:
                # 对节点 i: 受到来自 i+1 的拉力, 方向沿 dir_vec (指向 i+1) -> +F
                # 对节点 i+1: 受到来自 i 的拉力, 方向沿 -dir_vec (指向 i) -> -F
                f_vec = total_axial * dir_vec
                
                self.force[i] += f_vec
                self.force[i+1] -= f_vec

        # --- B. 外部作用力 (湿重, 水动力, 海床) ---
        for i in range(self.n_total_nodes):
            # 1. 湿重 (Wet Weight) W_i
            # 论文公式 3.45: W_{i+1/2} = 0.25 * pi * d^2 * l * (rho_w - rho) * g
            # 注意: rho_w - rho = fluid_den - line_den. 
            # 如果 line_den > fluid_den (钢), 则值为负, 代表方向向下(因为是rho_w-rho).
            # 本代码坐标系Z向上.
            # 这里的 W_i 是 "净" 力 (浮力 - 重力).
            # 计算单个节段的 (B - G):
            # Vol = A_sect * l
            # F_buoy_seg = rho_w * g * Vol
            # F_grav_seg = mass_den_linear * g * l
            # W_seg_net = F_buoy_seg - F_grav_seg (标量, 若重力大则为负)
            
            # [cite_start]节点 i 的湿重是相邻两节段的一半之和 [cite: 231]
            # 为简化, 假设所有节段相同 (端点已在质量中处理, 但力也应处理)
            factor = 1.0
            # 判断是否为端点 (简单起见，按质量比例分配)
            is_end = False
            for start, end in self.line_ranges:
                if i == start or i == end - 1:
                    is_end = True
                    break
            if is_end: factor = 0.5
            
            vol_seg = self.A_sect * self.l_seg0
            w_seg_net = (self.rho_w * vol_seg * self.g) - (self.mass_den_linear * self.l_seg0 * self.g)
            
            # W_i 沿 e_z 方向 (垂直)
            self.force[i, 2] += w_seg_net * factor
            
            # 2. 海底接触力 B_i
            # B_i = l * d * [ (Z_bot - Z_i)*k_b - Z_dot_i * c_b ] * e_z
            # 仅当 Z_i < Z_bot 时
            if self.pos[i, 2] < self.Z_bot:
                penetration = self.Z_bot - self.pos[i, 2] # > 0
                
                # 接触面积 A_contact = l * d (投影面积?) 公式中是 l d
                area_contact = self.l_seg0 * self.d * factor
                
                f_spring = penetration * self.kb
                f_damp = -self.vel[i, 2] * self.cb # 阻尼力方向与速度相反
                
                b_force = area_contact * (f_spring + f_damp)
                
                # 施加垂直力
                if b_force > 0: # 只能提供支撑力，不能拉住
                    self.force[i, 2] += b_force
                
                # 简单的水平摩擦 (论文公式中未明确写出水平摩擦项，但通常存在，保留原代码的小阻尼以防漂移)
                self.force[i, 0] -= 100.0 * self.vel[i, 0]
                self.force[i, 1] -= 100.0 * self.vel[i, 1]

            # [cite_start]3. 水动力 (Morison) [cite: 271]
            # 忽略波浪影响, 只考虑静水中缆运动
            # 相对速度 V_rel = 0 - V_node = -V_node
            v_node = self.vel[i]
            v_rel = -v_node 
            
            # 计算切向向量 q_i
            # q_i = (r_{i+1} - r_{i-1}) / |...|
            # 端点处理: 前向/后向差分
            q_i = np.zeros(3)
            
            # 查找当前节点属于哪根线以及位置
            # 简单处理: 根据存储的 line_ranges 加速查找
            # (在Python中循环查找较慢，但此处 N 较小)
            # 为性能优化，假设 i 的前后节点为 i-1, i+1 (需检查边界)
            
            idx_prev = i - 1
            idx_next = i + 1
            
            # 检查边界
            has_prev = True
            has_next = True
            for start, end in self.line_ranges:
                if i == start: has_prev = False
                if i == end - 1: has_next = False
            
            if has_prev and has_next:
                vec = self.pos[idx_next] - self.pos[idx_prev]
            elif has_next: # Start node
                vec = self.pos[idx_next] - self.pos[i]
            elif has_prev: # End node
                vec = self.pos[i] - self.pos[idx_prev]
            else:
                vec = np.array([0,0,1.0])
                
            norm_v = np.linalg.norm(vec)
            if norm_v > 1e-8:
                q_i = vec / norm_v
            else:
                q_i = np.array([0,0,1.0])
                
            # 分解相对速度
            # v_rel_tan = (v_rel dot q_i) * q_i
            # v_rel_norm = v_rel - v_rel_tan (即公式中的 (v dot q)q - v 的反向?)
            # 论文公式 3.51 定义: 
            # D_pi (横向) 使用 [(r_dot dot q)q - r_dot]. 这是 -V_normal
            # D_qi (切向) 使用 [(-r_dot) dot q] q. 这是 V_tangent
            
            val_tan = np.dot(v_rel, q_i)
            v_tan_vec = val_tan * q_i
            v_norm_vec = v_rel - v_tan_vec
            
            mag_tan = np.abs(val_tan)
            mag_norm = np.linalg.norm(v_norm_vec)
            
            # 拖曳力 D_pi, D_qi
            # D_pi = 0.5 * C_dn * rho_w * l * d * |v_n| * v_n
            # D_qi = 0.5 * C_dt * rho_w * pi * l * d * |v_t| * v_t (注意论文中有 pi)
            
            f_drag_norm = 0.5 * self.C_dn * self.rho_w * (self.l_seg0 * self.d * factor) * mag_norm * v_norm_vec
            f_drag_tan = 0.5 * self.C_dt * self.rho_w * (np.pi * self.l_seg0 * self.d * factor) * mag_tan * v_tan_vec
            
            self.force[i] += f_drag_norm + f_drag_tan
            
            # --- C. 求解动力学方程 ---
            # [m + a] r_ddot = F_total
            # m 是结构质量 (标量, 对角)
            # [cite_start]a 是附加质量矩阵 [a] = rho_w * A * l * [ C_an(I - q q^T) + C_at(q q^T) ] [cite: 286]
            
            # 构建节点质量矩阵 M_node (3x3)
            m_struct = self.node_mass[i] # scalar
            
            # 附加质量项系数
            coeff_add = self.rho_w * self.A_sect * (self.l_seg0 * factor)
            
            # 投影矩阵
            qqT = np.outer(q_i, q_i)
            I = np.eye(3)
            P_norm = I - qqT # 投影到法向平面
            
            # 附加质量矩阵 [a]
            M_added = coeff_add * (self.C_an * P_norm + self.C_at * qqT)
            
            # 总质量矩阵 [m + a]
            M_total = m_struct * I + M_added
            
            # 求解加速度: acc = M_total^-1 * Force
            # 对于固定节点 (锚点, 导缆孔在子步循环外已被强制赋值速度/位置), 
            # 但这里我们计算了力。为了避免覆盖边界条件, 仅对自由节点积分.
            is_fixed = False
            if i in self.anch_indices or i in self.fair_indices:
                is_fixed = True
            
            if not is_fixed:
                # 3x3 线性方程组求解
                acc = np.linalg.solve(M_total, self.force[i])
                
                # 积分 (Euler)
                self.vel[i] += acc * self.dt_physics
                self.pos[i] += self.vel[i] * self.dt_physics

    def close(self):
        pass