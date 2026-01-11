import numpy as np

class MoorEmmSolver:
    """
    MoorEmm: Python Native Lumped-Mass Mooring Solver
    文件名: moor_emm_solver.py
    
    基于集中质量法 (Lumped Mass Method) 的系泊动力学求解器。
    模拟 OpenFAST/MoorDyn 的物理原理，完全使用 Python/Numpy 实现。
    
    物理特性:
    - 离散化: 节点-弹簧-阻尼系统 (Node-Spring-Damper)
    - 受力: 轴向张力(EA), 湿重(重力-浮力), Morison水动力拖曳, 海底接触(弹簧阻尼)
    - 积分: 辛欧拉法 (Symplectic Euler) + 子步积分 (Sub-stepping) 以保证数值稳定性
    """
    def __init__(self, params, seastate, dt):
        print(f"[MoorEmm] Initializing Python Lumped-Mass Solver (moor_emm_solver)...")
        self.p = params
        self.seastate = seastate
        self.dt_global = float(dt)
        
        # --- 1. 物理参数解析 ---
        self.g = self.p.Gravity
        self.rho_w = self.p.WaterDensity
        self.depth = self.p.WaterDepth
        
        # 缆索属性 (假设所有缆索属性一致，基于 fast_params)
        self.L_total = self.p.Moor_LineLength
        self.mass_den = self.p.Moor_LineMass   # kg/m (空气中质量)
        self.EA = self.p.Moor_LineEA           # N (轴向刚度)
        self.Cd = self.p.Moor_DragCoeff        # 横向拖曳系数
        self.Ca = 1.0                          # 附加质量系数 (横向)
        
        # 估算等效直径 (用于水动力计算)
        # 假设链的体积密度约为钢 (7850 kg/m^3) -> mass_den = Area * 7850
        # Area = mass_den / 7850 -> Diam = 2 * sqrt(Area / pi)
        # 为了更通用的表现，这里采用近似反推，并设置最小值防止除零
        vol_per_m = self.mass_den / 7850.0 
        self.Diam = np.sqrt(vol_per_m / np.pi) * 2.0 
        if self.Diam < 0.05: self.Diam = 0.12 
        
        # --- 2. 离散化设置 ---
        self.n_lines = int(self.p.Moor_NumLines)
        self.n_segs = 10  # 每根缆索的分段数 (Python运行速度考虑，取10-20段)
        self.n_nodes_per_line = self.n_segs + 1
        self.l_seg0 = self.L_total / self.n_segs # 初始原长
        
        # 节点总数
        self.n_total_nodes = self.n_lines * self.n_nodes_per_line
        
        # --- 3. 初始化状态数组 (Vectorized) ---
        # Pos: (N, 3), Vel: (N, 3), Force: (N, 3), Mass: (N,)
        self.pos = np.zeros((self.n_total_nodes, 3))
        self.vel = np.zeros((self.n_total_nodes, 3))
        self.force = np.zeros((self.n_total_nodes, 3))
        self.mass = np.zeros(self.n_total_nodes)
        
        # 索引管理
        self.anch_indices = [] # 锚点索引 (Fixed)
        self.fair_indices = [] # 导缆孔索引 (Kinematic Coupling)
        self.line_ranges = []  # 每根缆索的节点范围 (start, end)
        
        self._initialize_geometry_and_mass()
        
        # --- 4. 求解器稳定性设置 (Sub-stepping) ---
        # 显式积分 Courant 条件: dt < L * sqrt(m / EA)
        # 计算临界时间步长
        m_node = self.mass[1] 
        wave_speed = np.sqrt(self.EA / (self.mass_den)) # Approx c = sqrt(E/rho)
        crit_dt = self.l_seg0 / wave_speed
        
        # 安全系数取 0.1 ~ 0.2
        self.dt_physics = min(0.005, crit_dt * 0.2) 
        self.n_substeps = int(np.ceil(self.dt_global / self.dt_physics))
        self.dt_physics = self.dt_global / self.n_substeps # 调整以匹配整步
        
        print(f"[MoorEmm] Setup: {self.n_lines} lines, {self.n_segs} segs/line.")
        print(f"[MoorEmm] Sub-stepping: {self.n_substeps} steps (dt_phys={self.dt_physics:.1e}s)")

    def _initialize_geometry_and_mass(self):
        """初始化节点位置(直线分布)与质量矩阵"""
        r_f = self.p.Moor_FairleadRadius
        z_f = -self.p.Moor_FairleadDraft
        r_a = self.p.Moor_AnchorRadius
        z_a = -self.p.Moor_AnchorDepth
        
        # 浮力与重力 (每米)
        # Buoyancy = rho_w * g * Vol_per_m
        area_hydro = np.pi * (self.Diam/2)**2
        buoyancy_per_m = area_hydro * self.rho_w * self.g
        weight_per_m = self.mass_den * self.g
        self.wet_weight_eff = weight_per_m - buoyancy_per_m # 净湿重(向下为正)
        
        seg_mass = self.l_seg0 * self.mass_den
        added_mass = self.l_seg0 * area_hydro * self.rho_w * self.Ca
        
        for i in range(self.n_lines):
            angle = i * (2*np.pi/self.n_lines) + np.pi
            
            # 锚点 (Node 0)
            p_anchor = np.array([r_a*np.cos(angle), r_a*np.sin(angle), z_a])
            # 导缆孔 (Node N)
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
                # 直线初始化 (简单，会在仿真开始几秒内快速下垂达到平衡)
                self.pos[idx] = (1 - frac) * p_anchor + frac * p_fair
                
                # 质量分配 (Lumped Mass)
                # 内部节点 = 1 seg mass; 端点 = 0.5 seg mass
                factor = 1.0
                if k == 0 or k == self.n_segs: factor = 0.5
                
                self.mass[idx] = (seg_mass + added_mass) * factor

    def update(self, state, t):
        """
        Input: state [surge, sway, heave, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        Output: F_mooring [Fx, Fy, Fz, Mx, My, Mz, ...] (14-array)
        """
        # 1. 平台运动解析
        plat_pos = np.array(state[0:3])
        plat_rot = np.array(state[3:6])
        plat_vel = np.array(state[6:9])
        plat_omega = np.array(state[9:12])
        
        # 旋转矩阵 (简化版: R = Rz * Ry * Rx)
        cx, sx = np.cos(plat_rot[0]), np.sin(plat_rot[0])
        cy, sy = np.cos(plat_rot[1]), np.sin(plat_rot[1])
        cz, sz = np.cos(plat_rot[2]), np.sin(plat_rot[2])
        
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        R = Rz @ Ry @ Rx
        
        # 2. 更新导缆孔目标位置 (边界条件)
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
            
        # 3. 物理子步循环 (Sub-stepping)
        for _ in range(self.n_substeps):
            # A. 强制设置边界条件 (Dirichlet BC)
            for k, idx in enumerate(self.fair_indices):
                self.pos[idx] = fair_targets[k]
                self.vel[idx] = fair_vels[k]
            
            for idx in self.anch_indices:
                self.vel[idx] = 0.0 # 锚点固定
            
            # B. 计算力 & 积分
            self._step_physics()

        # 4. 计算对平台的作用力 (Fairlead Tension)
        F_total = np.zeros(6)
        
        for k, idx in enumerate(self.fair_indices):
            # 取连接导缆孔的最后一节段
            prev_idx = idx - 1
            delta = self.pos[prev_idx] - self.pos[idx] # 指向缆索内部
            L = np.linalg.norm(delta)
            
            if L > 1e-6:
                # Tension = EA * strain
                strain = (L - self.l_seg0) / self.l_seg0
                if strain < 0: strain = 0
                tension = self.EA * strain
                
                f_vec = tension * (delta / L) # 作用在导缆孔上的力
                
                # Force Accumulation
                F_total[0:3] += f_vec
                # Moment Accumulation: r x F
                r_arm = self.pos[idx] - plat_pos
                F_total[3:6] += np.cross(r_arm, f_vec)
        
        # Return 14-element array (Standard LabFAST format)
        out = np.zeros(14)
        out[0:6] = F_total
        return out

    def _step_physics(self):
        """物理计算单步: Forces -> Acceleration -> Velocity -> Position"""
        self.force.fill(0.0)
        
        # 1. 湿重 (Gravity - Buoyancy)
        # z轴向下为负，重力向下。这里 wet_weight_eff = W - B (positive).
        # Force z = - (W - B)
        self.force[:, 2] -= self.wet_weight_eff * self.l_seg0 
        
        # 2. 弹性张力 (Spring Forces)
        for start, end in self.line_ranges:
            # 提取该线所有节点
            p_line = self.pos[start:end]
            # 向量: p[i+1] - p[i]
            delta = p_line[1:] - p_line[:-1]
            dists = np.linalg.norm(delta, axis=1)
            
            # Strain & Tension
            strains = (dists - self.l_seg0) / self.l_seg0
            strains[strains < 0] = 0 # No compression (Slack)
            T = self.EA * strains
            
            # Direction vectors
            with np.errstate(divide='ignore', invalid='ignore'):
                dirs = delta / dists[:, None]
                dirs[np.isnan(dirs)] = 0.0
            
            # f_seg 是弹簧力的大小方向
            f_seg = T[:, None] * dirs
            
            # 节点 i 被拉向 i+1 (+f_seg)
            self.force[start:end-1] += f_seg
            # 节点 i+1 被拉向 i (-f_seg)
            self.force[start+1:end] -= f_seg
            
        # 3. 水动力 (Morison Drag)
        # 获取每个节点的水质点速度 (假设 seastate 仅根据深度返回流速，忽略水平位置差异以提高速度)
        # 实际上应该调用 self.seastate.get_kinematics(x,y,z)，这里简化为循环
        for i in range(self.n_total_nodes):
            # 获取环境流速 (Wave + Current)
            u_fluid, _, _ = self.seastate.get_kinematics(self.pos[i,0], self.pos[i,1], self.pos[i,2])
            
            v_rel = u_fluid - self.vel[i]
            v_mag = np.linalg.norm(v_rel)
            
            if v_mag > 1e-5:
                # F_drag = 0.5 * rho * Cd * D * L * |v| * v
                f_d = 0.5 * self.rho_w * self.Cd * self.Diam * self.l_seg0 * v_mag * v_rel
                self.force[i] += f_d

        # 4. 海底接触 (Seabed Contact)
        # 简单的线性弹簧 + 阻尼地板模型
        k_bot = 1.0e4 # N/m
        c_bot = 1.0e3 # Ns/m
        
        # 找出触底节点 (z < -depth)
        bottom_indices = np.where(self.pos[:, 2] < -self.depth)[0]
        for i in bottom_indices:
            penetration = -self.depth - self.pos[i, 2] # > 0
            f_n = k_bot * penetration
            
            # 垂直阻尼
            v_z = self.vel[i, 2]
            if v_z < 0: # 向下运动(钻入)时阻尼更大
                f_n -= c_bot * v_z
                
            self.force[i, 2] += f_n
            
            # 简单水平摩擦 (减速)
            self.force[i, 0] -= 2000.0 * self.vel[i, 0]
            self.force[i, 1] -= 2000.0 * self.vel[i, 1]

        # 5. 时间积分 (Symplectic Euler)
        # 排除锚点和导缆孔 (由边界条件控制)
        free_nodes_mask = np.ones(self.n_total_nodes, dtype=bool)
        free_nodes_mask[self.anch_indices] = False
        free_nodes_mask[self.fair_indices] = False
        
        # a = F / m
        acc = self.force[free_nodes_mask] / self.mass[free_nodes_mask, None]
        
        # v = v + a * dt
        self.vel[free_nodes_mask] += acc * self.dt_physics
        # x = x + v * dt
        self.pos[free_nodes_mask] += self.vel[free_nodes_mask] * self.dt_physics

    def close(self):
        pass