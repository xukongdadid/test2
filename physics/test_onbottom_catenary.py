# -*- coding: utf-8 -*-
"""
test_onbottom_catenary.py  —  专门测试 catenary_moor.py 中 on-bottom 行为的小脚本

用法：
    1. 确保本文件与 catenary_moor.py 放在同一目录下；
    2. 运行：python test_onbottom_catenary.py
    3. 脚本会：
       - 构建一根单独的 CatenaryLine（Anchor 固定，Fairlead 在 x 方向做 surge）
       - 遍历一段 surge 范围，逐点调用 MAP++ 底部接触解析解 / 自由悬链线 / 兜底算法
       - 输出每个工况下的 H, V, L_B, |F|，并画出随 surge 变化的曲线
"""

import numpy as np
import matplotlib.pyplot as plt

from catenary_moor import CatenaryLine


# ----------------------------------------------------------------------
# 1. 构造一个简单的参数对象（仿照你的 params）
# ----------------------------------------------------------------------
class DummyParams:
    """
    简化版参数对象，只包含 CatenaryLine 需要的字段。
    你可以根据自己的项目把这些值改成真实参数。
    """
    # 线长 (m) —— 适当长一点，保证有 on-bottom 的可能
    Moor_LineLength = 900.0

    # 单位长度质量 (kg/m) —— 粗略取值
    Moor_LineMass = 800.0

    # 重力加速度
    Gravity = 9.81

    # 轴向刚度 EA (N) —— 取一个比较大的值（刚性较硬）
    Moor_LineEA = 1.0e9

    # 海底摩擦系数 C_B（关键参数，>0 才有 on-bottom 摩擦效应）
    Moor_CB = 1.0

    # 下面这两个在本测试脚本中不会真正用到，但 CatenaryLine.__init__ 里有调用
    Moor_NumLines = 1
    Moor_DragCoeff = 1.0

    # 其他 mooring 系统参数（在本脚本中都不会用到）
    Moor_FairleadRadius = 40.0
    Moor_FairleadDraft = 20.0
    Moor_AnchorRadius = 600.0
    Moor_AnchorDepth = 200.0


# ----------------------------------------------------------------------
# 2. 构造一根测试用的 CatenaryLine
# ----------------------------------------------------------------------
def build_test_line():
    params = DummyParams()

    # 设定一个简单几何：
    #   锚点在全局坐标系 (600, 0, -200)   —— 水深 200 m
    #   初始导缆孔在平台坐标系 (0, 0, -20)
    #   我们后面让平台在 x 方向 surge，因此 fairlead_global = [surge, 0, -20]
    anchor_pos = np.array([600.0, 0.0, -200.0])
    fairlead_local_pos = np.array([0.0, 0.0, -20.0])

    line = CatenaryLine(params, anchor_pos, fairlead_local_pos)
    return line


# ----------------------------------------------------------------------
# 3. 单点求解：给定 surge，返回 on-bottom / free 等详细信息
# ----------------------------------------------------------------------
def solve_one_offset(line, surge):
    """
    输入：
        line  : CatenaryLine 实例
        surge : 平台在 x 方向的位移（m），正方向为向 +X 方向移动

    输出：
        mode   : 'bottom' / 'free' / 'fallback'
        H      : 水平张力分量 (N)
        V      : 竖向张力分量 (N)
        LB     : 海底接触长度 (m)，若 mode != 'bottom' 则为 0 或 np.nan
        F_vec  : 作用在平台上的 3×1 力向量 (N)
    """
    # 锚点位置
    anchor = line.anchor.copy()

    # 导缆孔全局坐标（平台质心假定在原点）
    fairlead_global = np.array([surge, 0.0, -20.0])

    # 从锚点指向导缆孔的位移
    delta = fairlead_global - anchor
    X = np.sqrt(delta[0] ** 2 + delta[1] ** 2)  # 水平距离 l
    Z = delta[2]                                # 竖向高度差 h (向上为正)

    if X < 1.0e-3:
        X = 1.0e-3

    L = line.L
    w = line.w
    CB = line.CB
    L_min = np.sqrt(X * X + Z * Z)             # 线完全拉直时的长度

    H = V = None
    LB = 0.0
    mode = "unknown"

    # 1) 尝试 on-bottom 解析解
    if (L > 1.001 * L_min) and (CB > 0.0):
        Hb, Vb, LBb = line._solve_bottom_catenary_HLB(X, Z)
        if Hb is not None:
            H, V, LB = Hb, Vb, LBb
            mode = "bottom"

    # 2) 若没有成功，尝试自由悬链线
    if H is None or V is None:
        Hf, Vf = line._solve_free_catenary_HV(X, Z)
        if Hf is not None:
            H, V = Hf, Vf
            LB = 0.0
            mode = "free"

    # 3) 若仍失败，则兜底
    if H is None or V is None:
        H, V = line._fallback_solve_HV(X, Z)
        LB = np.nan
        mode = "fallback"

    # 4) 按 catenary_moor.py 的逻辑，把 (H, V) 转成“作用在平台上的力”
    # 水平单位向量：从锚点指向导缆孔在水平面上的方向
    dir_horiz = np.array([delta[0], delta[1], 0.0], dtype=float)
    norm_h = np.linalg.norm(dir_horiz)
    if norm_h < 1.0e-6:
        dir_horiz[:] = 0.0
    else:
        dir_horiz /= norm_h

    # 在局部线坐标中，H 指向导缆孔，V 向上；
    # 作用在平台上的力为反向：指向锚点 + 向下
    f_h = -H * dir_horiz
    f_v = np.array([0.0, 0.0, -V], dtype=float)
    F_vec = f_h + f_v

    return mode, H, V, LB, F_vec


# ----------------------------------------------------------------------
# 4. 主程序：扫描一段 surge，画曲线
# ----------------------------------------------------------------------
def main():
    line = build_test_line()

    # 设置 surge 扫描范围（可以根据自己需求修改）
    # 这里从 -50 m 到 +350 m，中间 81 个点
    surge_list = np.linspace(-50.0, 350.0, 81)

    modes = []
    H_list = []
    V_list = []
    LB_list = []
    Ft_list = []   # 总张力模 |F|
    Fx_list = []   # 水平力 Fx
    Fz_list = []   # 竖向力 Fz

    print("===== On-Bottom 测试结果（每个 surge 一个工况）=====")
    print("{:>8s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "surge", "mode", "H(N)", "V(N)", "LB(m)", "|F|(N)"))

    for s in surge_list:
        mode, H, V, LB, F_vec = solve_one_offset(line, s)

        F_norm = np.linalg.norm(F_vec)
        Fx, Fy, Fz = F_vec

        modes.append(mode)
        H_list.append(H)
        V_list.append(V)
        LB_list.append(LB)
        Ft_list.append(F_norm)
        Fx_list.append(Fx)
        Fz_list.append(Fz)

        print("{:8.2f} {:>10s} {:10.2f} {:10.2f} {:10.2f} {:10.2f}".format(
            s, mode, H, V, LB if not np.isnan(LB) else 0.0, F_norm))

    # 转成 numpy 数组，方便绘图
    surge_arr = np.array(surge_list)
    H_arr = np.array(H_list)
    V_arr = np.array(V_list)
    LB_arr = np.array(LB_list)
    Ft_arr = np.array(Ft_list)
    Fx_arr = np.array(Fx_list)
    Fz_arr = np.array(Fz_list)

    # ------------------------------------------------------------------
    # 绘图
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 8))

    # (1) 海底接触长度 L_B
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(surge_arr, LB_arr)
    ax1.set_ylabel("L_B (m)")
    ax1.set_title("On-bottom 行为测试：海底接触长度 L_B vs surge")
    ax1.grid(True)

    # (2) 水平 / 竖向张力 H, V
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(surge_arr, H_arr, label="H (horizontal)")
    ax2.plot(surge_arr, V_arr, label="V (vertical)")
    ax2.set_ylabel("Tension (N)")
    ax2.set_title("水平/竖向张力 vs surge")
    ax2.legend()
    ax2.grid(True)

    # (3) 作用在平台上的总力模 |F|
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(surge_arr, Ft_arr, label="|F| (total)")
    ax3.set_xlabel("surge (m)")
    ax3.set_ylabel("|F| (N)")
    ax3.set_title("平台上 mooring 总力 vs surge")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
