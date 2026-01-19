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
        # 数据来源: moordyn.dat
        # ==========================================
        
        # --- 1. 基础几何与结构参数 (Basic Geometry & Structure) ---
        self.Moor_NumLines = 3          # [数量] Line 1, 2, 3
        self.Moor_LineLength = 835.35   # [m] UnstrLen (无应力原长)
        self.Moor_LineMass = 113.35     # [kg/m] MassDen (线性密度)
        self.Moor_LineEA = 7.536e8      # [N] EA (轴向刚度)

        # --- 2. 水动力与截面参数 (Hydrodynamics & Cross-section) ---
        self.Moor_LineDiam = 0.0766     # [m] Diam (直径)
        
        # 拖曳系数 (Drag Coefficients)
        self.Moor_Cdn = 2.0             # [-] Cd (横向/法向拖曳系数)
        self.Moor_Cdt = 0.4             # [-] CdAx (切向/轴向拖曳系数)
        
        # 附加质量系数 (Added Mass Coefficients)
        self.Moor_Can = 0.8             # [-] Ca (横向/法向附加质量)
        self.Moor_Cat = 0.25            # [-] CaAx (切向/轴向附加质量)
        
        # 内部阻尼 (Internal Damping)
        # 注: moordyn.dat 中为 BA/-zeta = -1.0 (表示临界阻尼比), 
        # 但 MoorEmm 需要物理阻尼值 [N*s]. 
        # 对于较轻的 OC4 链条，1.0e8 过大，这里调整为 1.0e6 以避免刚性过大。
        self.Moor_Cint = 1.0e6          # [N*s] (估算值)

        # --- 3. 海床接触参数 (Seabed Contact) ---
        # 对应 moordyn.dat 中的 SOLVER OPTIONS
        self.Moor_Kb = 3.0e6            # [Pa/m] kbot (海床刚度)
        self.Moor_Cb = 3.0e5            # [Pa*s/m] cbot (海床阻尼)
        self.Moor_Fric_Mu_kT = 0      # [-] 动摩擦系数 (横向)
        self.Moor_Fric_Mu_kA = 0      # [-] 动摩擦系数 (轴向)
        self.Moor_Mc = 1              # 静摩擦/动摩擦比率
        self.Moor_Fric_Mu_sT = self.Moor_Fric_Mu_kT * self.Moor_Mc      # [-] 静摩擦系数 (横向)
        self.Moor_Fric_Mu_sA = self.Moor_Fric_Mu_kA * self.Moor_Mc      # [-] 静摩擦系数 (轴向)
        self.Moor_Fric_Cv = 1000.0      # [-] 摩擦线性区斜率参数

        # --- 3.5 离散化与求解设置 (Discretization & Solver Settings) ---
        self.Moor_NumSegs = 20          # [-] NumSegs (分段数)
        self.Moor_MaxDt = 0.001         # [s] dtM (物理求解最大步长)
        self.Moor_SubstepSafety = 0.8   # [-] 稳定性安全系数
        self.Moor_FallbackDt = 0.0005   # [s] 回退步长

        # --- 4. 导缆孔与锚点位置 (Fairlead & Anchor Geometry) ---
        # 数据来源: moordyn.dat "POINTS" 表
        # Line 1 连接 Anchor 1 和 Fairlead 4
        # Line 2 连接 Anchor 2 和 Fairlead 5
        # Line 3 连接 Anchor 3 和 Fairlead 6
        
        # 导缆孔坐标 (Vessel, 局部坐标)
        # [Line1_Fair(Node4), Line2_Fair(Node5), Line3_Fair(Node6)]
        self.Moor_FairleadPoints = [
            [ 20.434,  35.393, -14.0],  # Node 4
            [-40.868,   0.000, -14.0],  # Node 5
            [ 20.434, -35.393, -14.0]   # Node 6
        ]
        
        # 锚点坐标 (Fixed, 全局坐标)
        # [Line1_Anch(Node1), Line2_Anch(Node2), Line3_Anch(Node3)]
        self.Moor_AnchorPoints = [
            [ 418.8,  725.383, -200.0], # Node 1
            [-837.6,    0.000, -200.0], # Node 2
            [ 418.8, -725.383, -200.0]  # Node 3
        ]
        self.Moor_DragCoeff = 1.1
        # --- 5. 初始系泊模型选择 ---
        # 0 = Original Simple Solver (Quasi-static)
        # 1 = External Catenary / MAP++ 
        # 2 = MoorDyn (Dynamic Lumped-Mass, C++)
        # 3 = MoorEmm (Dynamic Lumped-Mass, Python Native)
        self.Moor_ModelType = 0
        # 海底摩擦系数 (用于 ModelType=1), 0.0 ~ 1.0
        self.Moor_CB = 0.5
        self.Moor_ModelPath = "model/PINN_DLCAL_v1.pt"

        # ==========================================
        # 可视化资源配置 (VTK 渲染)
        # ==========================================
        self.Visual_TurbineModelPath = ""  # 支持 .obj / .gltf / .glb
        self.Visual_SkyboxPath = ""        # HDRI/skybox 贴图路径 (.hdr/.png/.jpg)
        
        
        
        self.Init_Surge = 15.0  # 给 15米的初始偏移
        self.Init_Pitch = 5.0  # 或者给 5度的初始倾角
