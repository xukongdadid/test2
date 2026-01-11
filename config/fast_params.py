import numpy as np

class SystemParams:
    def __init__(self):
        # --- 几何与质量 (保持 V3.2) ---
        self.R = 63.0             
        self.HubHeight = 90.0     
        self.TowerHeight = 87.6   
        self.Mass_Blade = 17740.0 
        self.Mass_Nacelle = 240000.0 
        self.Mass_Hub = 56780.0
        self.Mass_Platform = 1.3447E+7 
        self.I_Roll = 6.827E+9
        self.I_Pitch = 6.827E+9
        self.I_Yaw = 1.226E+10
        # ✅ 新增：压载质量（按你自己 Parameters 里的数）
        self.Mass_Ballast = 3.5e5
        # ✅ 新增：塔顶等效质量（机舱 + 轮毂 + 三片叶片）
        self.Mtop = self.Mass_Nacelle + self.Mass_Hub + 3 * self.Mass_Blade
        # --- 环境: 风 (保持 V3.2) ---
        self.Env_WindType = 0       
        self.Env_WindSpeed = 11.4   
        self.Env_ShearExp = 0.2     
        self.Env_TurbIntensity = 10.0 
        self.Env_WindFile = ""      

        # --- [V3.3 Update] 环境: 波浪 ---
        # WaveMod: 0=Still, 1=Regular, 2=JONSWAP, 3=PM
        self.Env_WaveMod = 1        
        self.Env_WaveHeight = 3.66  # Hs (m)
        self.Env_WavePeriod = 9.7   # Tp (s)
        self.Env_WaveGamma = 3.3    # Peak Shape Parameter (JONSWAP)
        
        # --- [V3.3 New] 环境: 海流 (Current) ---
        # 叠加模型: U(z) = U_sub(z) + U_near(z) + U_ref
        self.Curr_RefSpeed = 0.0    # U_ref: 深度无关流 (m/s)
        self.Curr_RefDepth = 50.0   # 参考深度 (m)
        
        self.Curr_NearSurfSpeed = 0.0 # U_0_near: 近表层流速 (m/s, at z=0)
        # Linear profile to reference depth (0 at RefDepth)
        
        self.Curr_SubSpeed = 0.0      # U_0_sub: 次表层流速 (m/s, at z=0)
        self.Curr_SubExp = 0.143      # Power law exponent (1/7 typically)
        
        self.WaterDepth = 200.0      
        self.WaterDensity = 1025.0 
        self.AirDensity = 1.225    
        self.Gravity = 9.81        

        # --- [V3.3 New] 水动力建模方法 ---
        # 0=Strip Theory (Morison), 1=Potential Flow, 2=Hybrid
        self.Hydro_Method = 0 
        
        # OC4 几何参数
        self.Col_Diameter = 12.0   
        self.Cd_Col = 0.6          
        self.Ca_Col = 0.97

        self.RatedRotorSpeed = 12.1
        self.GearboxRatio = 97.0
        self.GenEfficiency = 0.944

        # ==========================================
        # 系泊系统参数配置 (Mooring System Configuration)
        # ==========================================
        # --- 1. 基础几何与结构参数 (Basic Geometry & Structure) ---
        self.Moor_NumLines = 3  # [数量] 系泊缆索的总根数 (通常为3根,呈120度对称分布)
        self.Moor_LineLength = 850  # [m]   缆索的无应力原长 (Unstretched Length, L0)
        self.Moor_LineMass = 685  # [kg/m] 缆索的线性密度 (干质量/单位长度), 用于计算重力与惯性
        self.Moor_LineEA = 3.27e9  # [N]   轴向拉伸刚度 (Axial Stiffness, E*A), 决定缆索弹性
        # --- 2. 水动力与截面参数 (Hydrodynamics & Cross-section) ---
        # [重要] 显式定义这些参数以覆盖 MoorEmmSolver 中的默认值
        self.Moor_LineDiam = 0.333  # [m]   缆索等效直径 (d). 用于计算浮力(体积)和水动力(面积)
        self.Moor_Cdn = 2  # [-]   法向拖曳系数 (Normal Drag Coeff, C_dn). 垂直于缆索的水流阻力
        self.Moor_Cdt = 0.4  # [-]   切向拖曳系数 (Tangential Drag Coeff, C_dt). 沿缆索表面的摩擦阻力
        self.Moor_Can = 0.82  # [-]   法向附加质量系数 (Added Mass Coeff, C_an). 也就是 Cm - 1
        self.Moor_Cat = 0.27  # [-]   (可选) 切向附加质量系数. 链条通常非0, 钢丝绳通常为0. 若不写默认为0.269(链条)
        self.Moor_Cint = 1.0e8  # [N*s/m^2] 缆索内部阻尼系数 (Internal Damping Coeff)
        # --- 3. 海床接触参数 (Seabed Contact) ---
        # 基于 Hall & Goupee (2015) Eq. 12 的接触模型
        self.Moor_Kb = 3.0e6  # [Pa/m] 或 [N/m^3] 海床垂直刚度 (Stiffness per unit area). 决定海床有多"硬"
        self.Moor_Cb = 3.0e5  # [Pa*s/m] 海床阻尼系数. 用于防止触底时的剧烈反弹 (如果不设置，代码通常有默认值)

        # --- 3.5 离散化与求解设置 (Discretization & Solver Settings) ---
        self.Moor_NumSegs = 20  # [-]   每根系泊线的离散节段数
        self.Moor_MaxDt = 0.00125  # [s]   物理子步最大步长上限
        self.Moor_SubstepSafety = 0.8  # [-]   稳定性安全系数 (用于临界步长)
        self.Moor_FallbackDt = 0.001  # [s]   无法计算临界步长时的回退步长

        # --- 4. 导缆孔与锚点位置 (Fairlead & Anchor Geometry) ---
        # 自定义导缆孔/锚点坐标 (必填)
        # - 导缆孔坐标为平台局部坐标系 (平台初始原点/姿态)
        # - 锚点坐标为全局坐标系 (海床固定)
        # - 列表长度需一致，且可与 Moor_NumLines 不同，求解器将以列表长度为准
        self.Moor_FairleadPoints = [[-58.0, 0.0, -14.0], [29.0, 50.229, -14.0], [29.0, -50.229, -14.0]]
        self.Moor_AnchorPoints = [[-837.6, 0.0, -200.0], [418.8, 725.383, -200.0], [418.8, -725.383, -200.0]]
        self.Moor_DragCoeff = 1.1
        # === 向后兼容 MoorEmmSolver 的“半径 + 吃水/水深”接口 ===
        # 导缆孔半径：按给定导缆孔坐标的水平距离取平均
        fair_r = [np.hypot(p[0], p[1]) for p in self.Moor_FairleadPoints]
        self.Moor_FairleadRadius = float(np.mean(fair_r))
        # 给的 z 是负的高度，吃水是正值
        self.Moor_FairleadDraft = -float(self.Moor_FairleadPoints[0][2])

        # 锚点半径 / 水深：同样从锚点坐标推回去
        anch_r = [np.hypot(p[0], p[1]) for p in self.Moor_AnchorPoints]
        self.Moor_AnchorRadius = float(np.mean(anch_r))
        self.Moor_AnchorDepth  = -float(self.Moor_AnchorPoints[0][2])

        # --- 5. 初始系泊模型选择 ---
        # 0 = Original Simple Solver (Quasi-static)
        # 1 = External Catenary / MAP++
        # 2 = MoorDyn (Dynamic Lumped-Mass, C++)
        # 3 = MoorEmm (Dynamic Lumped-Mass, Python Native)
        self.Moor_ModelType = 0
        # 海底摩擦系数 (用于 ModelType=1), 0.0 ~ 1.0
        self.Moor_CB = 0.5
