import torch
import numpy as np
from collections import deque
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

# 从当前文件位置一路向上找，直到找到包含 "aimodel" 文件夹的目录作为项目根目录
project_root = None
search_dir = current_dir
while True:
    if os.path.isdir(os.path.join(search_dir, "aimodel")):
        project_root = search_dir
        break
    parent = os.path.dirname(search_dir)
    if parent == search_dir:  # 到达磁盘根目录仍未找到
        break
    search_dir = parent

if project_root is None:
    raise ImportError(f"[Error] 未找到包含 aimodel/ 的项目根目录。current_dir={current_dir}")

# 把项目根目录加入 sys.path，确保可以 import aimodel.xxx
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # 确保 aimodel 文件夹在 sys.path 中 (代码前面已处理)，直接 import 模块名
    from aimodel.pinn_dacla_hallB import PINNDACLAHallB
except ImportError as e:
    print("\n[Error] 找不到 aimodel.pinn_dacla_hallB 或类 PINNDACLAHallB。")
    print("请确认文件存在: <project_root>/aimodel/pinn_dacla_hallB.py")
    print("并且文件内定义了: class PINNDACLAHallB")
    raise
# ===============================================================================

class PINNMooringWrapper:
    def __init__(self, params, seastate, dt):
        self.p = params
        self.seastate = seastate
        self.dt = dt
        
        # --- 1. 加载物理 Baseline ---
        from .mooring import InternalSimpleMooring
        self.phys_baseline = InternalSimpleMooring(params, seastate, dt)
        
        # --- 2. 加载训练好的模型 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # [修改点 1]：输入特征维度确认 = 18
        # 构成: Time(1) + Pos(6) + Vel(6) + EnvConsts(5) = 18
        in_features = 18 
        
        self.model = PINNDACLAHallB(
            in_features=in_features,
            out_dim=21, # 假设输出维度为 21 (Fairlead 3x3 + Anchor 3x3 + Extra 3)
            conv_channels=(64, 64, 128),
            lstm_hidden=128
        ).to(self.device)
        
        # 加载权重
        model_path = getattr(params, "Moor_ModelPath", "checkpoints/best_model.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            # 兼容处理 key 名称
            state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', None))
            if state_dict:
                self.model.load_state_dict(state_dict)
                print(f"PINN Model loaded from {model_path} (in_feat={in_features})")
            else:
                print("Error: Checkpoint structure unknown (keys: {checkpoint.keys()})")
        else:
            print(f"Error: Model path {model_path} not found!")
            
        self.model.eval()
        
        # --- 3. 数据处理准备 ---
        self.seq_len = 100 
        self.history_buffer = deque(maxlen=self.seq_len)
        
        # [修改点 2]：预读取 5 个环境常值 (根据您的描述)
        # 1. 风速 (使用仿真设定值)
        self.env_wind_spd = getattr(self.p, 'Env_WindSpeed', 0.0)
        # 2. 波高 (Hs)
        self.env_wave_hs = getattr(self.p, 'Env_WaveHeight', 0.0)
        # 3. 风谱周期 (这里取波浪周期 Tp)
        self.env_wave_tp = getattr(self.p, 'Env_WavePeriod', 0.0)
        # 4. 入流角 (Wave Heading / Direction) - 如果参数里没有则默认为 0
        self.env_wave_dir = getattr(self.p, 'Env_WaveDir', 0.0) 
        # 5. 水流流速 (Current Speed)
        self.env_curr_spd = getattr(self.p, 'Curr_RefSpeed', 0.0)

        print(f"PINN Environment Inputs: Wind={self.env_wind_spd}, Hs={self.env_wave_hs}, Tp={self.env_wave_tp}, Dir={self.env_wave_dir}, Curr={self.env_curr_spd}")

    def update(self, state, t):
        # state: 12维 [Surge, Sway, ..., Yaw, v_Surge, ..., v_Yaw]
        
        # 1. 物理模型计算 (Baseline)
        F_phys_ext = self.phys_baseline.update(state, t)
        
        # 2. 构造 18维特征向量
        # [0]: 时间
        # [1-6]: 位移 (Surge...Yaw)
        # [7-12]: 速度 (v_Surge...v_Yaw)
        # [13-17]: 5个环境常值
        
        pos = state[:6]
        vel = state[6:]
        env_feats = np.array([
            self.env_wind_spd, 
            self.env_wave_hs, 
            self.env_wave_tp, 
            self.env_wave_dir, 
            self.env_curr_spd
        ])
        
        # 拼接: [t] + pos(6) + vel(6) + env(5) = 18
        current_features = np.concatenate([[t], pos, vel, env_feats])
        
        # 3. 更新历史 buffer
        self.history_buffer.append(current_features)
        
        # 4. 历史不足时，直接返回物理结果
        if len(self.history_buffer) < self.seq_len:
            return F_phys_ext

        # 5. 推理
        x_seq_np = np.array(self.history_buffer) # (T, 18)
        
        # 构造物理引导输入 y_phys (需匹配模型定义的输入格式)
        # 这里简单将 6维物理力 扩展为 21维 (仅做格式对齐，具体映射需根据您的训练逻辑)
        y_phys_input = np.zeros(21)
        # 简单平均分配作为示例
        y_phys_input[:3] = F_phys_ext[:3] / 3.0 
        y_phys_input[3:6] = F_phys_ext[:3] / 3.0
        y_phys_input[6:9] = F_phys_ext[:3] / 3.0
        
        x_seq_tensor = torch.tensor(x_seq_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        y_phys_tensor = torch.tensor(y_phys_input, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(x_seq_tensor, y_phys_tensor, grl_lam=0.0)
            y_pred = out["y_pred"].cpu().numpy().squeeze() # (21,)
        
        # 6. 后处理：将 21维 预测力聚合为 6维 (Fx,Fy,Fz, Mx,My,Mz)
        F_total = self._map_model_output_to_force(y_pred)
        
        F_return = np.zeros(14)
        F_return[:6] = F_total
        return F_return

    def _map_model_output_to_force(self, y_pred):
        # 假设 y_pred 前9维是 Fairlead Force (3 lines * 3 xyz)
        forces = y_pred[:9].reshape(3, 3) 
        total_force = np.sum(forces, axis=0)
        
        # 力矩: 这里仅返回力，力矩暂设为0 (或者您可以使用物理模型的力矩)
        # 如果需要更精确，请根据导缆孔位置计算 R x F
        return np.concatenate([total_force, [0,0,0]])

    def close(self):
        pass