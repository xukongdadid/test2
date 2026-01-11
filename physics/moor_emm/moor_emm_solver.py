import numpy as np
from numba import njit


@njit(cache=True)
# ===核心物理模型模块===
def _solve_physics_jit(
        pos, vel, force,
        line_ranges, anch_indices, fair_indices,
        target_pos, target_vel,
        n_substeps, dt,
        g, rho_w, rho_mat, d, l_seg0, EA, C_int, Area,
        C_dn, C_dt, C_an, C_at, kb, cb, Z_bot
):
    """
    Numba 加速的物理核心求解器 (修正版)
    理论基础: Hall, M., & Goupee, A. (2015).
    """
    n_lines = line_ranges.shape[0]
    n_nodes = pos.shape[0]
    n_fairleads = len(fair_indices)

    # 定义全局坐标系的单位向量
    e_z = np.array([0.0, 0.0, 1.0])  # 垂直向上方向 (浮力方向)

    # ---------------------------
    # 子步循环 (Sub-stepping)
    # ---------------------------
    for _ in range(n_substeps):
        # 1. 强制边界条件 (Boundary Conditions)
        for i in range(len(anch_indices)):
            idx = anch_indices[i]
            vel[idx, :] = 0.0

        for k in range(n_fairleads):
            idx = fair_indices[k]
            pos[idx, :] = target_pos[k, :]
            vel[idx, :] = target_vel[k, :]

        # 2. 重置力数组
        force[:] = 0.0

        # 3. 节段循环 (Segment Loop): 计算所有基于“节段”的力
        # 包括: 内部刚度(Eq.3), 内部阻尼(Eq.5), 湿重分配(Eq.1 & 2)
        for line_idx in range(n_lines):
            start = line_ranges[line_idx, 0]
            end = line_ranges[line_idx, 1]

            for j in range(start, end - 1):
                # --- 几何信息 ---
                r_i = pos[j]
                r_ip1 = pos[j + 1]
                v_i = vel[j]
                v_ip1 = vel[j + 1]

                delta_r = r_ip1 - r_i
                dist = np.linalg.norm(delta_r)

                if dist < 1e-9: continue
                e_vec = delta_r / dist  # 节段轴向向量

                # --- A. 内部刚度与阻尼 (Internal Forces) ---
                # 轴向应变 epsilon (Eq. 3)
                epsilon = (dist / l_seg0) - 1.0  # 对应公式(5)工程应变

                # 张力 T (Eq. 3)
                tension = 0.0
                if epsilon > 0:
                    tension = EA * epsilon  # 对应公式(4)刚度张力,E、A为线材的弹性模量与截面积的乘积需要进行数值分离

                # 应变率 strain_rate (Eq. 6)
                delta_v = v_ip1 - v_i
                strain_rate = np.dot(delta_r, delta_v) / (l_seg0 * dist)

                # 阻尼力大小 C (Eq. 5)
                damping_mag = (C_int * Area) * strain_rate

                # 轴向力向量 (张力+阻尼)
                f_axial = (tension + damping_mag) * e_vec

                # --- B. 湿重/净浮力 (Wet Weight) ---
                # 修正: 在节段循环中计算，并分配给两端节点
                # 对应 Eq. 1: W_{i+1/2} = Area * l * (rho_w - rho_mat) * g
                # 注意:
                # 浮力 = + rho_w * V * g * e_z (向上)
                # 重力 = - rho_mat * V * g * e_z (向下)
                # 净力 = (rho_w - rho_mat) * Area * l_seg0 * g * e_z

                # 计算当前节段的总湿重 (向量)
                W_seg_mag = Area * l_seg0 * (rho_w - rho_mat) * g  # 对应技术报告公式(2)中的湿重标量
                W_seg_vec = W_seg_mag * e_z  # 显式乘以方向向量

                # --- 力分配 (Force Distribution) ---

                # 1. 轴向力 (作用力与反作用力)
                force[j] += f_axial
                force[j + 1] -= f_axial

                # 2. 湿重分配 (Eq. 2)
                # "This force is divided evenly among the two connecting nodes" [cite: 142]
                force[j] += 0.5 * W_seg_vec
                force[j + 1] += 0.5 * W_seg_vec

        # 4. 节点循环 (Node Loop): 计算基于“节点”的外部力
        # 包括: 海底接触(Eq.12), 水动力(Eq.8-9), 动力学求解
        for i in range(n_nodes):

            # --- C. 海底接触 (Bottom Contact) ---
            z_curr = pos[i, 2]
            if z_curr <= Z_bot:
                # 判断是否为端点来确定接触面积长度系数 (端点只有半个长度)
                is_endpoint = False
                for l_idx in range(n_lines):
                    if i == line_ranges[l_idx, 0] or i == line_ranges[l_idx, 1] - 1:
                        is_endpoint = True
                        break
                l_realline = l_seg0 * 0.5 if is_endpoint else l_seg0

                contact_area = d * l_realline
                penetration = Z_bot - z_curr
                z_v = vel[i, 2]

                # 垂向力 (Eq. 12)
                f_contact = contact_area * (penetration * kb - z_v * cb)
                if f_contact > 0:
                    force[i, 2] += f_contact

                # 水平摩擦 (防止数值漂移)
                # force[i, 0] -= 50.0 * vel[i, 0]
                # force[i, 1] -= 50.0 * vel[i, 1]

            # --- D. 水动力 (Hydrodynamics - Morison Eq) ---
            v_rel = -vel[i]  # 相对速度 (静水假设)

            # 估算切向向量 q_hat (寻找相邻节点)
            q_vec = np.zeros(3)
            # ... (查找逻辑保持不变) ...
            # 为节省篇幅，此处省略查找 q_vec 的代码，与上一版一致
            # 请保留原有的 q_vec 查找逻辑
            # -------------------------------------------------
            # 简单查找逻辑重述:
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
                    q_vec = pos[i + 1] - pos[i - 1]  # 中心差分
                elif has_next:
                    q_vec = pos[i + 1] - pos[i]  # 前向差分
                elif has_prev:
                    q_vec = pos[i] - pos[i - 1]  # 后向差分
            # -------------------------------------------------
            # 保护性编程，归一化得到 q_hat（单位切向向量）
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

            # 判断有效长度用于水动力 (端点减半)
            # 拖曳力
            f_drag_n = 0.5 * rho_w * C_dn * (d * l_realline) * norm_vn * v_norm_vec  # 横向力
            f_drag_t = 0.5 * rho_w * C_dt * (np.pi * d * l_realline) * norm_vt * v_tan_vec  # 切向力

            force[i] += f_drag_n + f_drag_t

            # --- E. 动力学方程求解 (Dynamics) ---
            # 仅对自由节点积分
            is_constrained = False
            for idx in anch_indices:
                if i == idx: is_constrained = True
            for idx in fair_indices:
                if i == idx: is_constrained = True

                m_struct = rho_mat * Area * l_realline
                qqT = np.outer(q_hat, q_hat)
                I3 = np.eye(3)

                M_added = rho_w * Area * l_realline * (C_an * (I3 - qqT) + C_at * qqT)
                M_struct = m_struct * I3
                M_total = M_struct + M_added

                # acc = M^-1 * F
                acc = np.linalg.solve(M_total, force[i])

                vel[i] += acc * dt
                pos[i] += vel[i] * dt


class MoorEmmSolver:
    """
    MoorEmm: Python Native Lumped-Mass Mooring Solver (Numba Accelerated)

    基于论文:
    Hall, M., & Goupee, A. (2015). Validation of a lumped-mass mooring line model...
    Ocean Engineering, 104, 590-603.
    """

    def __init__(self, params, seastate, dt):
        print(f"[MoorEmm] Initializing Hall & Goupee (2015) Solver (Numba Accelerated)...")
        self.p = params
        self.seastate = seastate
        self.dt_global = float(dt)
        self.custom_fairlead_points = None
        self.custom_anchor_points = None

        # --- 1. 物理参数解析 ---
        self.g = float(self.p.Gravity)
        self.rho_w = float(self.p.WaterDensity)
        self.depth = float(self.p.WaterDepth)

        self.d = float(self.p.Moor_LineDiam)
        self.mass_den_linear = float(self.p.Moor_LineMass)

        # 导出物理量
        self.Area = np.pi * self.d ** 2 / 4.0
        self.rho_mat = self.mass_den_linear / self.Area

        self.L_total = float(self.p.Moor_LineLength)
        self.EA = float(self.p.Moor_LineEA)
        self.C_int = float(self.p.Moor_Cint)

        # 水动力系数
        self.C_dn = float(self.p.Moor_Cdn)
        self.C_dt = float(self.p.Moor_Cdt)
        self.C_an = float(self.p.Moor_Can)
        self.C_at = float(self.p.Moor_Cat)

        # 海底接触参数
        self.kb = float(self.p.Moor_Kb)
        self.cb = float(self.p.Moor_Cb)
        self.Z_bot = -self.depth

        # --- 2. 离散化设置 ---
        self.n_lines = int(self.p.Moor_NumLines)
        self._load_custom_points()
        self.n_segs = int(self.p.Moor_NumSegs)
        self.n_nodes_per_line = self.n_segs + 1
        self.l_seg0 = self.L_total / self.n_segs

        self.n_total_nodes = self.n_lines * self.n_nodes_per_line

        # --- 3. 初始化状态 (使用 float64 数组) ---
        self.pos = np.zeros((self.n_total_nodes, 3), dtype=np.float64)
        self.vel = np.zeros((self.n_total_nodes, 3), dtype=np.float64)
        self.force = np.zeros((self.n_total_nodes, 3), dtype=np.float64)

        # 索引管理 (转换为 Numpy 数组以传递给 Numba)
        self.anch_indices_list = []
        self.fair_indices_list = []
        self.line_ranges_list = []

        self._initialize_geometry()

        # 转换为数组
        self.anch_indices = np.array(self.anch_indices_list, dtype=np.int64)
        self.fair_indices = np.array(self.fair_indices_list, dtype=np.int64)
        self.line_ranges = np.array(self.line_ranges_list, dtype=np.int64)

        # --- 4. 求解器时间步长 ---
        if self.mass_den_linear > 0:
            wave_speed = np.sqrt(self.EA / self.mass_den_linear)
            crit_dt = self.l_seg0 / wave_speed
            self.dt_physics = min(self.p.Moor_MaxDt, crit_dt * self.p.Moor_SubstepSafety)
        else:
            self.dt_physics = float(self.p.Moor_FallbackDt)

        self.n_substeps = int(np.ceil(self.dt_global / self.dt_physics))
        self.dt_physics = self.dt_global / self.n_substeps

        print(f"[MoorEmm] Physics: dt={self.dt_physics:.2e}s ({self.n_substeps} substeps).")

    def _load_custom_points(self):
        fairlead_points = getattr(self.p, "Moor_FairleadPoints", None)
        anchor_points = getattr(self.p, "Moor_AnchorPoints", None)
        if fairlead_points is None or anchor_points is None:
            raise ValueError("Moor_FairleadPoints and Moor_AnchorPoints must both be provided.")

        fairlead_points = np.asarray(fairlead_points, dtype=np.float64)
        anchor_points = np.asarray(anchor_points, dtype=np.float64)

        if fairlead_points.ndim != 2 or fairlead_points.shape[1] != 3:
            raise ValueError("Moor_FairleadPoints must be an N x 3 array-like.")
        if anchor_points.ndim != 2 or anchor_points.shape[1] != 3:
            raise ValueError("Moor_AnchorPoints must be an N x 3 array-like.")
        if fairlead_points.shape[0] != anchor_points.shape[0]:
            raise ValueError("Moor_FairleadPoints and Moor_AnchorPoints must have the same length.")
        if fairlead_points.shape[0] == 0:
            raise ValueError("Moor_FairleadPoints and Moor_AnchorPoints cannot be empty.")

        self.custom_fairlead_points = fairlead_points
        self.custom_anchor_points = anchor_points
        self.n_lines = int(fairlead_points.shape[0])

    def _initialize_geometry(self):
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
                self.pos[idx] = (1 - frac) * p_anchor + frac * p_fair

    def update(self, state, t):
        """
        主更新函数
        """
        # 1. 平台运动解析
        plat_pos = np.array(state[0:3], dtype=np.float64)
        plat_rot = np.array(state[3:6], dtype=np.float64)
        plat_vel = np.array(state[6:9], dtype=np.float64)
        plat_omega = np.array(state[9:12], dtype=np.float64)

        cx, sx = np.cos(plat_rot[0]), np.sin(plat_rot[0])
        cy, sy = np.cos(plat_rot[1]), np.sin(plat_rot[1])
        cz, sz = np.cos(plat_rot[2]), np.sin(plat_rot[2])
        R = np.array([[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                      [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                      [-sy, cy * sx, cy * cx]], dtype=np.float64)

        # 2. 计算导缆孔目标状态
        n_fair = len(self.fair_indices)
        target_pos = np.zeros((n_fair, 3), dtype=np.float64)
        target_vel = np.zeros((n_fair, 3), dtype=np.float64)

        for k in range(self.n_lines):
            p_local = self.custom_fairlead_points[k]

            target_pos[k] = plat_pos + R @ p_local
            target_vel[k] = plat_vel + np.cross(plat_omega, R @ p_local)

        # 3. 调用 JIT 物理求解器
        _solve_physics_jit(
            self.pos, self.vel, self.force,
            self.line_ranges, self.anch_indices, self.fair_indices,
            target_pos, target_vel,
            self.n_substeps, self.dt_physics,
            self.g, self.rho_w, self.rho_mat, self.d, self.l_seg0, self.EA, self.C_int, self.Area,
            self.C_dn, self.C_dt, self.C_an, self.C_at, self.kb, self.cb, self.Z_bot
        )

        # 4. 计算反作用力 (输出到 FAST)
        # 注意: 这里的力是最后一步的状态计算出来的
        F_total = np.zeros(6, dtype=np.float64)

        for k in range(n_fair):
            idx = self.fair_indices[k]
            # 连接导缆孔的内部节点是 idx - 1 (因为 fairlead 是 end_idx - 1)
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

    def close(self):
        pass
