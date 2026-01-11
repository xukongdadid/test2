#
# import matplotlib
# matplotlib.use('TkAgg')
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
#
# # 1. 導入外部模組
# from bodies import FlexibleBody, RigidBody
# from flexibility import GeneralizedMCK_PolyBeam
# # 注意：skew, point_jacobian 等工具現在改用您 v1.py 中的定義以保持完全一致
# from kinetics import skew
#
#
# # ============================================================
# # 平台工具函數 (保留自原 v1.py)
# # ============================================================
# def point_jacobian(s):
#     """線性小位移：點處位移 = [I, skew(s)] * q6"""
#     return np.hstack([np.eye(3), skew(s)])
#
#
# def assemble_global_K(points, K_locals):
#     K = np.zeros((6, 6))
#     for name, s in points.items():
#         J = point_jacobian(s)
#         K += J.T @ K_locals[name] @ J
#     return K
#
#
# # ============================================================
# # 參數與配置 (完全繼承自 v1.py)
# # ============================================================
# class Parameters:
#     def __init__(self):
#         self.Gravity = 9.80665
#         self.WaterDensity = 1025.0
#         self.Mass_Platform = 1.3447e7
#         self.Mass_Ballast = 3.5e5
#         self.Col_Diameter = 10.0
#         self.I_Roll = 6.827e9
#         self.I_Pitch = 6.827e9
#         self.I_Yaw = 1.226e10
#         # 塔架/機艙參數 (用於幾何剛度修正)
#         self.Mass_Nacelle = 2.4e5
#         self.Mass_Hub = 5.678e4
#         self.Mass_Blade = 1.774e4
#         self.Mtop = self.Mass_Nacelle + self.Mass_Hub + 3 * self.Mass_Blade
#
#
# class TowerConfig:
#     def __init__(self, p: Parameters):
#         self.H_tower = 87.6  # 繼承 v1.py 的高度
#         self.z_base = 10.0
#         self.nSpan = 50
#         self.s_span = np.linspace(0, self.H_tower, self.nSpan)
#         self.m = np.full(self.nSpan, 4000.0)  # 示例質量分布
#         self.EI = np.full(self.nSpan, 1e11)
#         self.coeff = np.array([[1.0, 0.0], [0.0, 1.0]])
#         self.exp = np.array([2, 3])
#         self.damp_zeta = np.array([0.01, 0.01])
#         self.Mtop = p.Mtop
#
#
# # ============================================================
# # 22DOF 全局結構模型
# # ============================================================
# class FloatingWindTurbineStructure:
#     def __init__(self, p: Parameters):
#         self.p = p
#
#         # --- 1. 先进行子系统建模，确定自由度数量 ---
#
#         # 塔架配置
#         t_cfg = TowerConfig(p)
#         n_tower_modes = len(t_cfg.damp_zeta)  # 获取塔架模态数 (例如 2)
#
#         # 动态设定总自由度：平台(6) + 塔架(n)
#         self.ndof = 6 + n_tower_modes
#
#         # 初始化矩阵
#         self.M = np.zeros((self.ndof, self.ndof))
#         self.C = np.zeros((self.ndof, self.ndof))
#         self.K = np.zeros((self.ndof, self.ndof))
#
#         # -------------------------------------------------
#         # 2. 平台建模 (0:6)
#         # -------------------------------------------------
#         self.m_plat = p.Mass_Platform + p.Mass_Ballast
#         self.I_plat = np.diag([p.I_Roll, p.I_Pitch, p.I_Yaw])
#
#         A_wp = np.pi * (p.Col_Diameter / 2.0) ** 2 * 4.0
#         V_disp = self.m_plat / p.WaterDensity
#         Draft = V_disp / A_wp
#         Z_G = -0.5 * Draft
#
#         # 刚体质量 + 附加质量
#         plat_rigid = RigidBody(name='Platform', mass=self.m_plat, J=self.I_plat, s_OG=[0, 0, Z_G])
#         M6_added = np.diag([0.15 * self.m_plat, 0.15 * self.m_plat, 0.35 * self.m_plat, 1.5e9, 1.5e9, 1.5e11])
#         M6 = plat_rigid.mass_matrix + M6_added  # 注意：这里不要加括号 ()
#
#         # 刚度 K6
#         K6 = np.zeros((6, 6))
#         K6[2, 2] = p.WaterDensity * p.Gravity * A_wp
#         k_rp = p.WaterDensity * p.Gravity * V_disp * abs(Z_G)
#         K6[3, 3] = k_rp
#         K6[4, 4] = k_rp
#
#         # 系泊
#         R = 40.0;
#         zF = -15.0;
#         asymmetry_eps = 0.03
#         ang = np.deg2rad([0.0, 120.0, 240.0])
#         pts = {f'F{i}': np.array([R * np.cos(a), R * np.sin(a), zF]) for i, a in enumerate(ang)}
#         k_h = 5.0e4;
#         k_v = 1.0e5
#         Kloc = {}
#         for i, (name, s) in enumerate(pts.items()):
#             er = np.array([s[0], s[1], 0.0]);
#             er /= np.linalg.norm(er)
#             ez = np.array([0.0, 0.0, 1.0])
#             scale = (1.0 + asymmetry_eps) if i == 0 else 1.0
#             Kloc[name] = (scale * k_h) * np.outer(er, er) + k_v * np.outer(ez, ez)
#
#         K6 += assemble_global_K(pts, Kloc)
#         K6[5, 5] += 5.0e7
#
#         # 阻尼 C6
#         zeta = 0.02
#         omega_ref = np.sqrt(K6[2, 2] / M6[2, 2])
#         C6 = (zeta * omega_ref) * M6 + (zeta / omega_ref) * K6
#         C6[5, 5] += 1.0e6
#
#         # 填入全局矩阵
#         self.M[:6, :6] = M6
#         self.K[:6, :6] = K6
#         self.C[:6, :6] = C6
#
#         # -------------------------------------------------
#         # 3. 塔架建模 (6:end)
#         # -------------------------------------------------
#         # 计算属性
#         props = GeneralizedMCK_PolyBeam(
#             s_span=t_cfg.s_span, m=t_cfg.m, EIFlp=t_cfg.EI, EIEdg=t_cfg.EI,
#             coeffs=t_cfg.coeff, exp=t_cfg.exp,
#             damp_zeta=t_cfg.damp_zeta, gravity=p.Gravity, Mtop=t_cfg.Mtop
#         )
#
#         # 实例化 FlexibleBody
#         self.tower_body = FlexibleBody(name='Tower', r_O=[0, 0, t_cfg.z_base])
#         # 手动注入矩阵
#         self.tower_body.MM = props['MM']
#         self.tower_body.KK = props['KK']
#         self.tower_body.DD = props['DD']
#
#         # 组装到全局矩阵 (利用切片操作，自动适应 ndof)
#         # 塔架本身的 DOFs 从索引 6 开始直到结束
#         self.M[6:, 6:] = self.tower_body.MM[6:, 6:]
#         self.K[6:, 6:] = self.tower_body.KK[6:, 6:]
#         self.C[6:, 6:] = self.tower_body.DD[6:, 6:]
#
#         # 耦合项 (平台 <-> 塔架)
#         self.M[:6, 6:] = self.tower_body.MM[:6, 6:]
#         self.M[6:, :6] = self.tower_body.MM[6:, :6]
#
#     def rhs(self, t, y):
#         q = y[:self.ndof]
#         v = y[self.ndof:]
#         # 现在 M 是满秩的 (8x8)，可以求逆
#         a = np.linalg.solve(self.M, -self.C @ v - self.K @ q)
#         return np.hstack([v, a])
#
#
# # ============================================================
# # 示例運行
# # ============================================================
# if __name__ == "__main__":
#     params = Parameters()
#     fwt = FloatingWindTurbineStructure(params)
#
#     # 动态获取自由度数量
#     ndof = fwt.ndof
#     print(f"系统总自由度: {ndof} (6 Platform + {ndof - 6} Tower)")
#
#     q0 = np.zeros(ndof)  # 自动匹配大小 (例如 8)
#     q0[2] = 2.0  # Heave
#     q0[4] = np.deg2rad(3.0)  # Pitch
#
#     # 状态向量长度为 2 * ndof
#     y0 = np.hstack([q0, np.zeros(ndof)])
#
#     sol = solve_ivp(fwt.rhs, [0, 100], y0, t_eval=np.linspace(0, 100, 1000))
#
#     plt.figure()
#     plt.plot(sol.t, sol.y[4] * 180 / np.pi, label='Platform Pitch (deg)')
#     # 如果有塔架模态，绘制第一个模态 (索引 6)
#     if ndof > 6:
#         plt.plot(sol.t, sol.y[6], label='Tower Mode 1 (m)')
#
#     plt.legend();
#     plt.grid(True);
#     plt.show()
import numpy as np
import matplotlib

# 强制使用 TkAgg 后端，修复 PyCharm 报错
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 导入外部模组
from bodies import FlexibleBody, RigidBody
from flexibility import GeneralizedMCK_PolyBeam
# 注意：使用 kinetics 中的工具
from kinetics import skew


# ============================================================
# 平台工具函数
# ============================================================
def point_jacobian(s):
    """线性小位移：点处位移 = [I, skew(s)] * q6"""
    return np.hstack([np.eye(3), skew(s)])


def assemble_global_K(points, K_locals):
    K = np.zeros((6, 6))
    for name, s in points.items():
        J = point_jacobian(s)
        K += J.T @ K_locals[name] @ J
    return K


# ============================================================
# 参数与配置
# ============================================================
class Parameters:
    def __init__(self):
        self.Gravity = 9.80665
        self.WaterDensity = 1025.0
        self.Mass_Platform = 1.3447e7
        self.Mass_Ballast = 3.5e5
        self.Col_Diameter = 10.0
        self.I_Roll = 6.827e9
        self.I_Pitch = 6.827e9
        self.I_Yaw = 1.226e10
        # 塔架/机舱参数 (用于几何刚度修正)
        self.Mass_Nacelle = 2.4e5
        self.Mass_Hub = 5.678e4
        self.Mass_Blade = 1.774e4
        self.Mtop = self.Mass_Nacelle + self.Mass_Hub + 3 * self.Mass_Blade


class TowerConfig:
    def __init__(self, p: Parameters):
        self.H_tower = 87.6
        self.z_base = 10.0
        self.nSpan = 50
        self.s_span = np.linspace(0, self.H_tower, self.nSpan)
        self.m = np.full(self.nSpan, 4000.0)
        self.EI = np.full(self.nSpan, 1e11)
        self.coeff = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.exp = np.array([2, 3])
        self.damp_zeta = np.array([0.01, 0.01])
        self.Mtop = p.Mtop


# ============================================================
# 浮式风机结构类 (动态 DOF)
# ============================================================
class FloatingWindTurbineStructure:
    def __init__(self, p: Parameters):
        self.p = p

        # --- 1. 确定系统自由度 ---
        t_cfg = TowerConfig(p)
        n_tower_modes = len(t_cfg.damp_zeta)
        self.ndof = 6 + n_tower_modes  # 动态计算 (6+2=8)

        self.M = np.zeros((self.ndof, self.ndof))
        self.C = np.zeros((self.ndof, self.ndof))
        self.K = np.zeros((self.ndof, self.ndof))

        # -------------------------------------------------
        # 2. 平台建模 (0:6)
        # -------------------------------------------------
        self.m_plat = p.Mass_Platform + p.Mass_Ballast
        self.I_plat = np.diag([p.I_Roll, p.I_Pitch, p.I_Yaw])

        # 水静力
        A_wp = np.pi * (p.Col_Diameter / 2.0) ** 2 * 4.0
        V_disp = self.m_plat / p.WaterDensity
        Draft = V_disp / A_wp
        Z_G = -0.5 * Draft

        # 质量矩阵 (修复: mass_matrix 不加括号)
        plat_rigid = RigidBody(name='Platform', mass=self.m_plat, J=self.I_plat, s_OG=[0, 0, Z_G])
        M6_added = np.diag([0.15 * self.m_plat, 0.15 * self.m_plat, 0.35 * self.m_plat, 1.5e9, 1.5e9, 1.5e11])
        M6 = plat_rigid.mass_matrix + M6_added

        # 刚度矩阵
        K6 = np.zeros((6, 6))
        K6[2, 2] = p.WaterDensity * p.Gravity * A_wp
        k_rp = p.WaterDensity * p.Gravity * V_disp * abs(Z_G)
        K6[3, 3] = k_rp
        K6[4, 4] = k_rp

        # 系泊刚度 (含破对称)
        # R = 40.0;
        # zF = -15.0;
        # asymmetry_eps = 0.03
        # ang = np.deg2rad([0.0, 120.0, 240.0])
        # pts = {f'F{i}': np.array([R * np.cos(a), R * np.sin(a), zF]) for i, a in enumerate(ang)}
        # k_h = 5.0e4;
        # k_v = 1.0e5
        # Kloc = {}
        # for i, (name, s) in enumerate(pts.items()):
        #     er = np.array([s[0], s[1], 0.0]);
        #     er /= np.linalg.norm(er)
        #     ez = np.array([0.0, 0.0, 1.0])
        #     scale = (1.0 + asymmetry_eps) if i == 0 else 1.0
        #     Kloc[name] = (scale * k_h) * np.outer(er, er) + k_v * np.outer(ez, ez)
        #
        # K6 += assemble_global_K(pts, Kloc)
        k_h = 5.0e4  # 径向刚度
        k_v = 1.0e5  # 垂向刚度
        k_t = 1.0e4  # 切向刚度 (修复：确保定义了它)

        # 缆绳几何定义
        R = 40.0;
        zF = -15.0
        asymmetry_eps = 0.05
        ang = np.deg2rad([0.0, 120.0, 240.0])
        pts = {f'F{i}': np.array([R * np.cos(a), R * np.sin(a), zF]) for i, a in enumerate(ang)}

        Kloc = {}

        for i, (name, s) in enumerate(pts.items()):
            # 1. 向量计算
            er = np.array([s[0], s[1], 0.0])
            er /= np.linalg.norm(er)  # 径向单位向量
            ez = np.array([0.0, 0.0, 1.0])  # 垂向单位向量
            et = np.cross(ez, er)  # 切向单位向量

            # 2. 破对称系数计算
            # 这里的写法是安全的，无论 i 是几，scale 都会被赋值
            scale = (1.0 + asymmetry_eps) if i == 1 else 1.0

            # 3. 刚度矩阵赋值 (关键！！！)
            # 注意：这行代码必须与上面的 er, ez, et, scale 保持同一缩进层级
            # 绝对不要把它放在 if/else 里面
            Kloc[name] = (scale * k_h) * np.outer(er, er) + \
                         k_v * np.outer(ez, ez) + \
                         (scale * k_t) * np.outer(et, et)

        # 循环结束后再组装
        K6 += assemble_global_K(pts, Kloc)
        K6[5, 5] += 5.0e7

        # 阻尼矩阵
        zeta = 0.02
        omega_ref = np.sqrt(K6[2, 2] / M6[2, 2])
        C6 = (zeta * omega_ref) * M6 + (zeta / omega_ref) * K6
        C6[5, 5] += 1.0e6

        self.M[:6, :6] = M6
        self.K[:6, :6] = K6
        self.C[:6, :6] = C6

        # -------------------------------------------------
        # 3. 塔架建模 (6:end)
        # -------------------------------------------------
        # 修复: 参数名匹配 (EIFlp, EIEdg, coeffs)
        props = GeneralizedMCK_PolyBeam(
            s_span=t_cfg.s_span,
            m=t_cfg.m,
            EIFlp=t_cfg.EI,
            EIEdg=t_cfg.EI,
            coeffs=t_cfg.coeff,
            exp=t_cfg.exp,
            damp_zeta=t_cfg.damp_zeta,
            gravity=p.Gravity,
            Mtop=t_cfg.Mtop
        )

        # 修复: FlexibleBody 初始化 + 矩阵注入
        self.tower_body = FlexibleBody(name='Tower', r_O=[0, 0, t_cfg.z_base])
        self.tower_body.MM = props['MM']
        self.tower_body.KK = props['KK']
        self.tower_body.DD = props['DD']

        # 组装 (利用切片自动适应 ndof)
        # 塔架内部项
        self.M[6:, 6:] = self.tower_body.MM[6:, 6:]
        self.K[6:, 6:] = self.tower_body.KK[6:, 6:]
        self.C[6:, 6:] = self.tower_body.DD[6:, 6:]

        # 耦合项 (Platform <-> Tower)
        self.M[:6, 6:] = self.tower_body.MM[:6, 6:]
        self.M[6:, :6] = self.tower_body.MM[6:, :6]

    def rhs(self, t, y):
        q = y[:self.ndof]
        v = y[self.ndof:]
        a = np.linalg.solve(self.M, -self.C @ v - self.K @ q)
        return np.hstack([v, a])


