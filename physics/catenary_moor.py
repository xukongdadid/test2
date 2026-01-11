# -*- coding: utf-8 -*-
"""
catenary_moor.py  —  MAP++ 风格悬链线系泊模块 (含自由悬链线 + 底部接触解析解)

参考：MAP++ Documentation, Release 1.15, Chapter 3 Theory (Free–Hanging Line & Line Touching the Bottom)

核心特点：
1. 单根锚链使用 MAP++ 理论中的闭式解析公式：
   - 3.1.1 自由悬链线 (Free–Hanging Line)：未知 H, V，通过 l, h 两个方程联立求解。
   - 3.1.2 底部接触 (Line Touching the Bottom)：未知 H, L_B（海底接触长度），
     V 由 V = ω (L - L_B) 给出，再通过 l, h 两个方程联立求解。
   - 其中 ω = w 为浸没重度，EA 为轴向刚度，C_B 为海底摩擦系数。
2. 自动判断并优先尝试“底部接触”解（如果线足够长且 C_B > 0）；
   若求解失败或接触长度 L_B≈0，则退回到自由悬链线解；
   若仍失败，则最后回退到一个一维近似算法，保证数值稳健。
3. 与原 mooring.py 保持同样的外部接口：
   - MooringSystem(params, seastate)
   - update(state, t) -> F_ext (长度 14，前 6 为 mooring 对平台的广义力)
"""

import numpy as np
from scipy.optimize import root, root_scalar


class CatenaryLine:
    """
    单根悬链线求解器 (MAP++ 风格解析 + 底部接触)
    """

    def __init__(self, params, anchor_pos, fairlead_local_pos):
        self.p = params
        self.anchor = np.array(anchor_pos, dtype=float)       # 锚点全局坐标
        self.fairlead_local = np.array(fairlead_local_pos, dtype=float)  # 导缆孔在平台坐标系下位置

        # 线参数
        self.L = float(params.Moor_LineLength)                # 总长 L
        self.w = float(params.Moor_LineMass) * float(params.Gravity)  # 线的浸没重度 ω = w (近似)
        self.EA = float(params.Moor_LineEA)                   # 轴向刚度 EA

        # 海底摩擦系数 C_B，如果没有此参数，则默认为 0（不考虑底摩擦）
        self.CB = float(getattr(params, "Moor_CB", 0.0))

        # 数值安全阈值
        self._H_min = 1.0e2
        self._H_max = 1.0e9

    # ------------------------------------------------------------------
    #  1. 自由悬链线 (MAP++ 3.1.1) —— 以 H, V 为未知
    # ------------------------------------------------------------------
    def _free_catenary_equations(self, HV, L, w, EA, l, h):
        """
        自由悬链线闭式方程（MAP++ 3.1.1），未知 H, V：
            l = H/ω [ ln(V/H + sqrt(1+(V/H)^2)) - ln((V-ωL)/H + sqrt(1+((V-ωL)/H)^2)) ] + HL/EA
            h = H/ω [ sqrt(1+(V/H)^2) - sqrt(1+((V-ωL)/H)^2) ] + 1/EA (V L - ω L^2/2)

        这里直接用 Va = V - ω L 的等价形式。
        """
        H, V = HV
        # 保证 H > 0
        H = max(H, self._H_min)

        Va = V - w * L  # 垂向锚点力分量（MAP++ 中的 V_a）

        H_inv = 1.0 / H
        A = (Va + w * L) * H_inv   # = V/H
        B = Va * H_inv             # = (V - ωL)/H

        def _log_term(X):
            return np.log(X + np.sqrt(1.0 + X * X))

        # 水平位移 l_model
        l_model = (H / w) * (_log_term(A) - _log_term(B)) + (H * L) / EA

        # 竖向位移 h_model
        z_term = np.sqrt(1.0 + A * A) - np.sqrt(1.0 + B * B)
        h_model = (H / w) * z_term + (1.0 / EA) * (Va * L + 0.5 * w * L * L)

        return np.array([l_model - l, h_model - h], dtype=float)

    def _solve_free_catenary_HV(self, l, h):
        """
        求解自由悬链线的 H, V。
        若成功，返回 (H, V)；若失败，返回 (None, None)。
        """
        L = self.L
        w = self.w
        EA = self.EA

        # 初始猜测：H0 ~ 0.5 wL, V0 ~ wL
        H0 = max(0.5 * w * L, self._H_min)
        V0 = w * L

        HV0 = np.array([H0, V0], dtype=float)

        try:
            sol = root(
                self._free_catenary_equations,
                HV0,
                args=(L, w, EA, l, h),
                method="hybr",
                tol=1e-8,
            )
        except Exception:
            return None, None

        if not sol.success:
            return None, None

        H, V = sol.x
        if H < self._H_min or not np.isfinite(H) or not np.isfinite(V):
            return None, None

        return float(H), float(V)

    # ------------------------------------------------------------------
    #  2. 底部接触线 (MAP++ 3.1.2) —— 以 H, L_B 为未知，V = ω (L - L_B)
    # ------------------------------------------------------------------
    def _bottom_catenary_equations(self, HLb, L, w, EA, CB, l, h):
        """
        底部接触（on-bottom）情况下的闭式方程（MAP++ 3.1.2）：

        已知：l（水平总位移）、h（竖向位移）、L, w, EA, C_B
        未知：H, L_B
        其中 V = ω (L - L_B)，x0 为水平张力由 0 过渡到正值的位置。

        MAP++ 给出的闭式公式：
            λ = { L_B - H / (C_B ω)   if x0 > 0
                { 0                  otherwise

            l = L_B + (H/ω) ln[ V/H + sqrt(1 + (V/H)^2) ]
                + H L / EA + C_B ω / (2 EA) ( x0 λ - L_B^2 )

            h = (H/ω) [ sqrt(1 + (V/H)^2) - 1 ] + V^2 / (2 EA ω)
        """
        H, LB = HLb

        # 保证变量范围
        H = max(H, self._H_min)
        LB = float(np.clip(LB, 0.0, L))

        if CB <= 0.0:
            # 无摩擦时不适用该模型，返回大残差
            return np.array([1e6, 1e6], dtype=float)

        # 悬挂段的垂向力：V = ω (L - L_B)
        V = w * (L - LB)

        # x0 与 λ
        x0_raw = LB - H / (CB * w)  # 由 T_e(s) = MAX[H + C_B ω (s - L_B), 0] 得到
        if x0_raw > 0.0:
            x0 = x0_raw
            lam = x0_raw  # λ = L_B - H/(C_B ω) = x0
        else:
            x0 = 0.0
            lam = 0.0

        # l 公式
        VH = V / H
        log_arg = VH + np.sqrt(1.0 + VH * VH)
        if log_arg <= 0:
            # 避免 log(非正)
            return np.array([1e6, 1e6], dtype=float)

        l_model = (
            LB
            + (H / w) * np.log(log_arg)
            + (H * L) / EA
            + (CB * w / (2.0 * EA)) * (x0 * lam - LB * LB)
        )

        # h 公式
        h_model = (H / w) * (np.sqrt(1.0 + VH * VH) - 1.0) + (V * V) / (2.0 * EA * w)

        return np.array([l_model - l, h_model - h], dtype=float)

    def _solve_bottom_catenary_HLB(self, l, h):
        """
        求解底部接触线的 H, L_B。
        若成功返回 (H, V, L_B)，若失败返回 (None, None, None)。

        注意：
        - 只有当 C_B > 0 且线“足够长”时才有意义。
        - V 总是通过 V = ω (L - L_B) 得到。
        """
        L = self.L
        w = self.w
        EA = self.EA
        CB = self.CB

        if CB <= 0.0:
            return None, None, None

        # 线几何下限：完全拉紧时的长度
        L_min = np.sqrt(l * l + h * h)
        if L <= 1.001 * L_min:
            # 线不够长，不会有明显海底接触
            return None, None, None

        # 使用自由悬链线解作为初始猜测
        Hf, Vf = self._solve_free_catenary_HV(l, h)
        if Hf is None:
            H0 = max(0.5 * w * L, self._H_min)
            # 先假定有一半长度悬挂
            LB0 = 0.5 * L
        else:
            H0 = Hf
            # 由 V = ω (L - L_B) 反推 L_B
            LB0 = float(np.clip(L - Vf / w, 0.0, L))

        HLb0 = np.array([H0, LB0], dtype=float)

        try:
            sol = root(
                self._bottom_catenary_equations,
                HLb0,
                args=(L, w, EA, CB, l, h),
                method="hybr",
                tol=1e-8,
            )
        except Exception:
            return None, None, None

        if not sol.success:
            return None, None, None

        H, LB = sol.x
        if H < self._H_min or LB < 0.0 or LB > L or not np.isfinite(H):
            return None, None, None

        # 计算 V
        V = w * (L - LB)

        # 要求至少有一点海底接触，否则认为没有真正 on-bottom
        if LB < 1.0e-3:
            return None, None, None

        return float(H), float(V), float(LB)

    # ------------------------------------------------------------------
    #  3. 最后兜底：原 mooring.py 的一维近似算法
    # ------------------------------------------------------------------
    def _fallback_solve_HV(self, l, h):
        """
        若 MAP++ 自由悬链线 + 底部接触解都失败，使用原 mooring.py 中的简化算法作为兜底。
        这不是 MAP++ 理论的一部分，只是为了防止数值发散导致全局仿真崩溃。
        """
        L = self.L
        w = self.w
        EA = self.EA

        Z_target = h

        def span_mismatch(H):
            if H <= 1.0:
                H = 1.0
            term = 2.0 * H / w
            val = Z_target * (Z_target + term)
            if val < 0.0:
                val = 0.0
            L_s = np.sqrt(val)
            if L_s >= L:
                L_s = L

            x_s = (H / w) * np.arcsinh(w * L_s / H)
            x_b = L - L_s
            elastic_stretch = (H * L) / EA
            return (x_s + x_b + elastic_stretch) - l

        try:
            sol = root_scalar(
                span_mismatch,
                bracket=[self._H_min, self._H_max],
                method="brentq",
                xtol=0.1,
            )
            H = sol.root
        except Exception:
            H = 1.0e5

        term = 2.0 * H / w
        val = Z_target * (Z_target + term)
        if val < 0.0:
            val = 0.0
        L_s = np.sqrt(val)
        if L_s > L:
            L_s = L
        V = w * L_s

        return float(H), float(V)

    # ------------------------------------------------------------------
    #  4. 对外接口：给定导缆孔全局坐标，返回作用在平台上的力
    # ------------------------------------------------------------------
    def solve_force(self, fairlead_global):
        """
        输入：
        - fairlead_global : 导缆孔在全局坐标系中的位置 3×1

        输出：
        - f_line : 作用在平台上的 3×1 力向量 (N)，方向为“锚链对平台”的作用力。
        """
        fairlead_global = np.array(fairlead_global, dtype=float)

        # 从锚点指向导缆孔的位移（局部悬链线坐标）
        delta = fairlead_global - self.anchor
        X = np.sqrt(delta[0] ** 2 + delta[1] ** 2)  # 水平距离 l
        Z = delta[2]                                # 竖向高度差 h (向上为正)

        if X < 1.0e-3:
            X = 1.0e-3

        # 线完全拉直时的最小长度
        L_min = np.sqrt(X * X + Z * Z)

        H, V = None, None

        # --- 4.1 优先尝试 MAP++ 底部接触解析解（前提：线比几何长度长且有摩擦） ---
        if (self.L > 1.001 * L_min) and (self.CB > 0.0):
            Hb, Vb, LB = self._solve_bottom_catenary_HLB(X, Z)
            if Hb is not None:
                H, V = Hb, Vb

        # --- 4.2 若没有成功，再尝试自由悬链线解 ---
        if H is None or V is None:
            Hf, Vf = self._solve_free_catenary_HV(X, Z)
            if Hf is not None:
                H, V = Hf, Vf

        # --- 4.3 若仍失败，则使用兜底近似算法 ---
        if H is None or V is None:
            H, V = self._fallback_solve_HV(X, Z)

        # 水平单位向量：从锚点指向导缆孔在水平面上的方向
        dir_horiz = np.array([delta[0], delta[1], 0.0], dtype=float)
        norm_h = np.linalg.norm(dir_horiz)
        if norm_h < 1.0e-6:
            dir_horiz[:] = 0.0
        else:
            dir_horiz /= norm_h

        # 在局部线坐标中，H 指向导缆孔，V 向上；
        # 作用在平台上的力应为反向：指向锚点 + 向下
        f_h = -H * dir_horiz
        f_v = np.array([0.0, 0.0, -V], dtype=float)

        return f_h + f_v


class MooringSystem:
    """
    WOUSE / WOGPT 悬链线系泊系统 (MAP++ 理论内核)

    与原 mooring.MooringSystem 保持同样的接口：
    - __init__(params, seastate)
    - update(state, t) -> F_ext (长度 14 的数组，前 6 个为 mooring 对平台的广义力 [Fx, Fy, Fz, Mx, My, Mz])
    """

    def __init__(self, params, seastate):
        self.p = params
        self.seastate = seastate
        self.lines = []
        self.fairlead_points = None
        self.anchor_points = None
        self._setup_lines()

    def _load_custom_points(self):
        fairlead_points = getattr(self.p, "Moor_FairleadPoints", None)
        anchor_points = getattr(self.p, "Moor_AnchorPoints", None)
        if fairlead_points is None or anchor_points is None:
            raise ValueError("Moor_FairleadPoints and Moor_AnchorPoints must both be provided.")

        fairlead_points = np.asarray(fairlead_points, dtype=float)
        anchor_points = np.asarray(anchor_points, dtype=float)

        if fairlead_points.ndim != 2 or fairlead_points.shape[1] != 3:
            raise ValueError("Moor_FairleadPoints must be an N x 3 array-like.")
        if anchor_points.ndim != 2 or anchor_points.shape[1] != 3:
            raise ValueError("Moor_AnchorPoints must be an N x 3 array-like.")
        if fairlead_points.shape[0] != anchor_points.shape[0]:
            raise ValueError("Moor_FairleadPoints and Moor_AnchorPoints must have the same length.")
        if fairlead_points.shape[0] == 0:
            raise ValueError("Moor_FairleadPoints and Moor_AnchorPoints cannot be empty.")

        self.fairlead_points = fairlead_points
        self.anchor_points = anchor_points

    # ------------------------------------------------------------------
    #  1. 根据参数定义多条锚链的几何布置
    # ------------------------------------------------------------------
    def _setup_lines(self):
        """
        使用自定义导缆孔/锚点坐标列表：
        - Moor_FairleadPoints : 导缆孔坐标 (平台局部坐标系)
        - Moor_AnchorPoints   : 锚点坐标 (全局坐标系)
        """
        self._load_custom_points()
        n = int(self.fairlead_points.shape[0])
        self.lines = []
        for i in range(n):
            # 导缆孔在平台坐标系中的位置
            fl = np.array(self.fairlead_points[i], dtype=float)

            # 锚点在全局坐标系中的位置（假定初始平台在原点）
            an = np.array(self.anchor_points[i], dtype=float)

            self.lines.append(CatenaryLine(self.p, an, fl))

    # ------------------------------------------------------------------
    #  2. 主接口：根据当前平台状态计算 mooring 力
    # ------------------------------------------------------------------
    def update(self, state, t):
        """
        输入：
        - state : 平台状态向量，假定包含
            state[0:3] : [surge, sway, heave]
            state[3:6] : [roll, pitch, yaw]
            state[6:9] : [surge_dot, sway_dot, heave_dot] (用于拖曳计算，可选)

        - t     : 当前时间 (s)，暂未使用，仅为兼容接口。

        输出：
        - F_ext : 长度 14 的数组，前 6 个为 mooring 对平台的广义力 [Fx, Fy, Fz, Mx, My, Mz]。
        """
        surge, sway, heave = state[0], state[1], state[2]
        roll, pitch, yaw = state[3], state[4], state[5]

        # 平台质心位置（假定初始在原点）
        plat_pos = np.array([surge, sway, heave], dtype=float)

        # 小角度近似旋转矩阵（与原 mooring.py 保持一致）
        R = np.eye(3)
        R[0, 1] = -yaw
        R[0, 2] = pitch
        R[1, 0] = yaw
        R[1, 2] = -roll
        R[2, 0] = -pitch
        R[2, 1] = roll

        # 总广义力 [Fx, Fy, Fz, Mx, My, Mz]
        F_total = np.zeros(6, dtype=float)

        # 平均流速，用于简化拖曳力
        v_curr_avg = self.seastate.get_current_velocity(-100.0)

        # 平台平动速度
        if len(state) >= 9:
            v_plat = np.array([state[6], state[7], state[8]], dtype=float)
        else:
            v_plat = np.zeros(3, dtype=float)

        rho_water = 1025.0  # 海水密度
        Cd = float(self.p.Moor_DragCoeff)
        D_eff = 0.2         # 线的等效直径，用于拖曳面积估算

        for line in self.lines:
            # 导缆孔全局位置
            fl_global = plat_pos + R @ line.fairlead_local

            # (1) 悬链线张力（包含底部接触 + 自由悬链线解析解）
            f_elastic = line.solve_force(fl_global)

            # (2) 线拖曳力（相对流速）
            v_rel = np.array(v_curr_avg, dtype=float) - v_plat
            v_rel_norm = np.linalg.norm(v_rel)
            if v_rel_norm > 1.0e-6:
                A_eff = D_eff * line.L
                f_drag = 0.5 * rho_water * Cd * A_eff * v_rel * v_rel_norm
            else:
                f_drag = np.zeros(3, dtype=float)

            # 减弱竖向拖曳
            f_drag[2] *= 0.1

            # 该线总力
            f_line_total = f_elastic + f_drag

            # 累加到总力
            F_total[0:3] += f_line_total

            # 力矩：r × F
            r_vec = fl_global - plat_pos
            F_total[3:6] += np.cross(r_vec, f_line_total)

        # 按原接口构造 F_ext（长度 14，前 6 个为 mooring 广义力）
        F_ext = np.zeros(14, dtype=float)
        F_ext[0:6] = F_total

        return F_ext
