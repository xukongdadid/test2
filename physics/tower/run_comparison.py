# ============================================================
# run_comparison.py
# ------------------------------------------------------------
# 功能：对比 OpenFAST .out 文件与 Python Kane 模型的仿真结果
# 适配：structure_kane_method_v1.py (FloatingWindTurbineStructure)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os

# ============================================================
# 1. 导入您的模型
# ============================================================
# 确保 structure_kane_method_v1.py 在同一目录下
try:
    from structure_kane_method_v1 import Parameters, FloatingWindTurbineStructure
except ImportError:
    print("❌ 错误：找不到 structure_kane_method_v1.py，请确认文件名或路径。")
    exit()


# ============================================================
# 2. 读取 OpenFAST .out 文件
# ============================================================
def read_openfast_out(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    with open(path, "r") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        # OpenFAST 输出文件通常以 "Time" 开头作为数据头
        if line.strip().startswith("Time"):
            header_idx = i
            break

    if header_idx is None:
        raise RuntimeError("❌ 未找到 OpenFAST .out 表头（Time ...）")

    columns = lines[header_idx].split()
    data = []

    # 从表头后两行开始读取（跳过单位行）
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        # 跳过包含括号的单位行，如 "(s) (m) ..."
        if "(" in line or ")" in line:
            continue

        try:
            row = [float(x) for x in line.split()]
            if len(row) == len(columns):
                data.append(row)
        except ValueError:
            continue

    return pd.DataFrame(data, columns=columns)


# ============================================================
# 3. 运行 Python 仿真 (封装逻辑)
# ============================================================
def run_python_simulation(t_end, dt=0.025):
    """
    运行 Kane 模型并返回结果
    """
    print(f"🚀 正在运行 Python Kane 模型 (Tmax={t_end}s)...")

    # 1. 初始化模型
    p = Parameters()
    fwt = FloatingWindTurbineStructure(p)

    # 2. 设置初始条件 (与 OpenFAST 保持一致)
    ndof = fwt.ndof
    q0 = np.zeros(ndof)
    q0[2] = 2.0  # Heave [m]
    q0[4] = np.deg2rad(3.0)  # Pitch [rad] (注意：Python内计算用弧度)

    # 状态向量 [q, v]
    y0 = np.hstack([q0, np.zeros(ndof)])

    # 3. 时间序列
    t_eval = np.arange(0, t_end, dt)

    # 4. 求解
    sol = solve_ivp(fwt.rhs, [0, t_end], y0, t_eval=t_eval, method='RK45')

    return sol.t, sol.y


# ============================================================
# 4. 主程序
# ============================================================
def main():
    # --------------------------------------------------------
    # 配置：OpenFAST 文件路径
    # --------------------------------------------------------
    FAST_OUT_FILE = r"C:\Users\谢丞尧\Desktop\LAB Hu研究生\ELASTO\IEA-15-240-RWT-1.1\IEA-15-240-RWT-1.1.17\OpenFAST\对比\IEA-15-240-RWT-UMaineSemi.out"

    # --------------------------------------------------------
    # 配置：通道映射与单位转换
    # 格式: (图表标题, OpenFAST列名, Python索引, Python单位转换因子)
    # --------------------------------------------------------
    # OpenFAST 的角度通常是 Deg，Python 算出来是 Rad，所以要 * 180/pi
    dof_map = [
        ("Surge", "PtfmSurge", 0, 1.0),  # m -> m
        ("Sway", "PtfmSway", 1, 1.0),  # m -> m
        ("Heave", "PtfmHeave", 2, 1.0),  # m -> m
        ("Roll", "PtfmRoll", 3, 180 / np.pi),  # rad -> deg
        ("Pitch", "PtfmPitch", 4, 180 / np.pi),  # rad -> deg
        ("Yaw", "PtfmYaw", 5, 180 / np.pi)  # rad -> deg
    ]

    # 1. 读取 OpenFAST 数据
    try:
        print(f"📂 读取 OpenFAST 文件: {FAST_OUT_FILE}")
        df_fast = read_openfast_out(FAST_OUT_FILE)
        t_fast = df_fast["Time"].values
    except Exception as e:
        print(e)
        return

    # 2. 运行 Python 仿真 (使用与 OpenFAST 相同的时长)
    t_py, y_py_raw = run_python_simulation(t_end=t_fast[-1], dt=0.025)

    # 3. 绘图对比
    # 创建 3x2 的子图布局，一次性显示所有自由度
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    print("📊 正在绘图...")

    for i, (name, fast_col, py_idx, scale) in enumerate(dof_map):
        ax = axes[i]

        # --- OpenFAST 数据 ---
        if fast_col in df_fast.columns:
            y_fast = df_fast[fast_col].values
            ax.plot(t_fast, y_fast, 'k-', label='OpenFAST', linewidth=1.5, alpha=0.7)
        else:
            print(f"⚠️ 警告: OpenFAST 文件中缺少列 {fast_col}")

        # --- Python 数据 (插值对齐) ---
        # 获取原始数据并进行单位转换 (例如 rad -> deg)
        y_py_data = y_py_raw[py_idx] * scale

        # 简单线性插值以便在同一横坐标下对比 (可选)
        f_interp = interp1d(t_py, y_py_data, kind='linear', fill_value="extrapolate")
        y_py_interp = f_interp(t_fast)

        ax.plot(t_fast, y_py_interp, 'r--', label='Python Kane', linewidth=1.5)

        # 样式设置
        ax.set_title(f"Platform {name} Response")
        ax.set_xlabel("Time (s)")

        # 根据物理量设置 Y 轴标签
        if name in ["Roll", "Pitch", "Yaw"]:
            ax.set_ylabel("Angle (deg)")
        else:
            ax.set_ylabel("Displacement (m)")

        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()