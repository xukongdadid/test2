# physics/structure_kane_v1.py
import numpy as np

from physics.tower.bodies import RigidBody
from physics.tower.kinetics import skew         # 你自己的 kinetics.py 里已经有
from physics.tower.flexibility import GeneralizedMCK_PolyBeam


def point_jacobian(s):
    """
    线性小位移假设下，点的位移 = [I, skew(s)] * q6
    s: 点相对平台参考点 O 的位置向量 (3,)
    返回: 3x6 Jac
    """
    return np.hstack([np.eye(3), skew(s)])


def assemble_global_K(points, K_locals):
    """
    从多个局部 3x3 刚度矩阵 K_locals[name] 组装成平台参考点 O 的 6x6 刚度矩阵。
    points[name] 是该力作用点在平台坐标系下的位置向量 s (3,)
    """
    K = np.zeros((6, 6))
    for name, s in points.items():
        J = point_jacobian(s)
        K += J.T @ K_locals[name] @ J
    return K


class TowerConfig:
    """
    塔架模态配置（简单版本）
    - 目前只给出 2 个模态：FA1, SS1
    - 后面你可以改成 4 模态 (FA1, SS1, FA2, SS2) 只要把 coeff/exp/damp_zeta 扩一下
    """
    def __init__(self, p):
        self.H_tower = 87.6
        self.z_base = 10.0
        self.nSpan = 50

        # 沿高度方向的节点
        self.s_span = np.linspace(0, self.H_tower, self.nSpan)

        # 质量和刚度分布（先用均匀占位，你后续可从 ElastoDyn 读）
        self.m = np.full(self.nSpan, 4000.0)
        self.EI = np.full(self.nSpan, 1.0e11)

        # 模态多项式系数和幂次（这里给 2 个模态）
        # 形如 U(s) = sum_k coeff[k,mode] * (s/H)^exp[k]
        self.coeff = np.array([[1.0, 0.0],
                               [0.0, 1.0]])
        self.exp   = np.array([2, 3])

        # 对应模态阻尼比
        self.damp_zeta = np.array([0.01, 0.01])

        # 塔顶集中质量 = 机舱 + 轮毂 + 三片叶片
        self.Mtop = (
            p.Mass_Nacelle
            + p.Mass_Hub
            + 3.0 * p.Mass_Blade
        )


class KaneStructureModel:
    """
    新版结构模型：
    - 使用 Kane + 模态梁的 M, C, K
    - 目前结构自由度数设为 8：
        0..5: 平台 6DOF [Surge, Sway, Heave, Roll, Pitch, Yaw]
        6..7: 塔架前 2 个模态 [TwrFA1, TwrSS1] （示意）
      对应 state[0:8], state[22:30]
    - 全局 state 长度仍为 44，结构只作用在前 8 个 DOF 上，其余 DOF 先保持不变
    """
    def __init__(self, params):
        self.p = params

        # --- 1. 结构 DOF 数 ---
        self.ndof_struct = 8          # 6 平台 + 2 模态
        self.ndof_total  = 44         # 整个 state 长度

        # 全局 M, C, K 里只用一个 8x8 子块
        self.M = np.zeros((self.ndof_struct, self.ndof_struct))
        self.C = np.zeros((self.ndof_struct, self.ndof_struct))
        self.K = np.zeros((self.ndof_struct, self.ndof_struct))

        # --- 2. 平台 6DOF 质量/刚度/阻尼 (0:6) ---
        p = self.p
        m_plat = p.Mass_Platform + p.Mass_Ballast
        I_plat = np.diag([p.I_Roll, p.I_Pitch, p.I_Yaw])

        # 水线面积 (假设四个立柱，和你 v1 里的写法一致)
        A_wp = np.pi * (p.Col_Diameter / 2.0) ** 2 * 4.0
        V_disp = m_plat / p.WaterDensity
        Draft  = V_disp / A_wp
        Z_G    = -0.5 * Draft

        # 刚体平台 + 附加质量
        plat_rigid = RigidBody(
            name='Platform',
            mass=m_plat,
            J=I_plat,
            s_OG=[0.0, 0.0, Z_G]
        )

        # 经验附加质量（你 v1 里的那组数）
        M6_added = np.diag([
            0.15 * m_plat,
            0.15 * m_plat,
            0.35 * m_plat,
            1.5e9,
            1.5e9,
            1.5e11
        ])
        M6 = plat_rigid.mass_matrix + M6_added

        # 水静力刚度
        K6 = np.zeros((6, 6))
        # Heave
        K6[2, 2] = p.WaterDensity * p.Gravity * A_wp
        # Roll/Pitch 恢复力矩
        k_rp = p.WaterDensity * p.Gravity * V_disp * abs(Z_G)
        K6[3, 3] = k_rp
        K6[4, 4] = k_rp

        # 系泊刚度（示意，复制你 v1 中的 3 根对称布置）
        R   = 40.0
        zF  = -15.0
        asym_eps = 0.03
        ang = np.deg2rad([0.0, 120.0, 240.0])
        pts = {f'F{i}': np.array([R*np.cos(a), R*np.sin(a), zF]) for i, a in enumerate(ang)}

        k_h = 5.0e4
        k_v = 1.0e5
        Kloc = {}
        for i, (name, s) in enumerate(pts.items()):
            er = np.array([s[0], s[1], 0.0])
            er /= np.linalg.norm(er)
            ez = np.array([0.0, 0.0, 1.0])
            scale = (1.0 + asym_eps) if i == 0 else 1.0
            Kloc[name] = (scale * k_h) * np.outer(er, er) + k_v * np.outer(ez, ez)

        K6 += assemble_global_K(pts, Kloc)
        # 给 Yaw 一点刚度，防止奇异
        K6[5, 5] += 5.0e7

        # Rayleigh 阻尼（以 Heave 为参考）
        zeta       = 0.02
        omega_ref  = np.sqrt(K6[2, 2] / M6[2, 2])
        C6         = (zeta * omega_ref) * M6 + (zeta / omega_ref) * K6
        C6[5, 5]  += 1.0e6

        # 填入结构矩阵的 0:6,0:6 块
        self.M[:6, :6] = M6
        self.K[:6, :6] = K6
        self.C[:6, :6] = C6

        # --- 3. 塔架模态 2DOF (6:8) ---
        t_cfg = TowerConfig(p)

        props = GeneralizedMCK_PolyBeam(
            s_span   = t_cfg.s_span,
            m        = t_cfg.m,
            EIFlp    = t_cfg.EI,
            EIEdg    = t_cfg.EI,
            coeffs   = t_cfg.coeff,
            exp      = t_cfg.exp,
            damp_zeta= t_cfg.damp_zeta,
            gravity  = p.Gravity,
            Mtop     = t_cfg.Mtop
        )
        # props['MM'], ['KK'], ['DD'] 是 (6+nm)×(6+nm) 的矩阵
        MM = props['MM']
        KK = props['KK']
        DD = props['DD']

        # 注意：MM 的前 6×6 是平台刚体 + 塔架对平台的惯性耦合
        #      后面部分是塔架模态 DOF
        # 这里只取出包含平台+2 模态的子块 (8×8)
        # [修复后代码 START] =======================================
        # 1. 叠加质量矩阵 (Mass)
        # 将 "平台+压载+附加质量" (M6) 与 "塔架贡献" (MM) 相加
        # M6 是 6x6，MM 是 8x8 (假设 ndof=8)

        # 先把 M6 填进去 (基础)
        self.M[:6, :6] = M6

        # 再把塔架的 MM 叠加上去 (注意是 +=)
        # MM[:8, :8] 的前 6x6 是塔架对底座的耦合，后 2x2 是塔架模态
        self.M[:8, :8] += MM[:8, :8]

        # 2. 叠加刚度矩阵 (Stiffness)
        # K6 包含静水恢复力 + 系泊刚度
        self.K[:6, :6] = K6
        # KK 包含塔架弹性刚度
        self.K[:8, :8] += KK[:8, :8]

        # 3. 叠加阻尼矩阵 (Damping)
        self.C[:6, :6] = C6
        self.C[:8, :8] += DD[:8, :8]
        # [修复后代码 END] =========================================

        # --- 4. 基本健壮性：确保 M 可逆 ---
        # 如果你后面把参数改得比较激进，建议打印一下 eig(M) 检查
        if np.linalg.cond(self.M) > 1e12:
            print("[WARN] KaneStructureModel: M is ill-conditioned, check parameters.")
            # --- 4.b 修补 Yaw DOF 的惯性，防止 M 奇异 ---
            # 对应 DOF 顺序: [Surge, Sway, Heave, Roll, Pitch, Yaw, TwrFA1, TwrSS1]
        if abs(self.M[5, 5]) < 1e-6:
                # 用系统参数里的 I_Yaw 做兜底惯量
            self.M[5, 5] = self.p.I_Yaw

    def get_derivative(self, state, F_gen_total, rotor_speed=None):
        """
        state: 长度 44
            state[0:22]   : 22 个结构位移 DOF（你现在定义的那套）
            state[22:44]  : 22 个结构速度 DOF
        F_gen_total: 长度 22，对应那 22 个结构 DOF 的广义外力
        我们目前只在前 8 个 DOF 上使用新的 M,C,K，其余 DOF 先不动。
        """
        # --- 1. 拆 q, qd ---
        q  = state[:22]
        qd = state[22:44]

        # 只取前 8 个 DOF 作为“结构核心”
        q_s   = q[:self.ndof_struct]
        qd_s  = qd[:self.ndof_struct]
        F_s   = F_gen_total[:self.ndof_struct]

        # --- 2. 结构核心的加速度:  M q̈ + C q̇ + K q = F ---
        qdd_s = np.linalg.solve(self.M, F_s - self.C @ qd_s - self.K @ q_s)

        # --- 3. 组装 dydt ---
        dydt = np.zeros_like(state)
        # 位移导数 = 速度
        dydt[:22] = qd
        # 前 8 个 DOF 的加速度来自 Kane 模型，后面的先设为 0
        dydt[22:22 + self.ndof_struct] = qdd_s

        # 其余 DOF（比如高阶塔架模态、转子扭转、叶片 DOF）先不动：
        # 如果你后续要接入更完整的 22DOF 模型，可以在这里扩展。

        return dydt
