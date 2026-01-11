import numpy as np

class SeaState:
    """
    WOUSE V4.1 SeaState Module (OpenFAST Architecture)
    --------------------------------------------------
    职责:
    1. 统一管理海洋环境参数 (波浪、海流)
    2. 生成随机波浪场 (JONSWAP/PM Spectrum)
    3. 计算空间任意点 (x,y,z) 在任意时刻 (t) 的流体运动学 (u,v,w, ax,ay,az, Pressure)
    4. 不涉及力学计算，只提供运动学数据
    """
    def __init__(self, params):
        self.p = params
        self.g = self.p.Gravity
        self.rho = self.p.WaterDensity
        self.depth = self.p.WaterDepth
        self.time = 0.0
        
        # 波浪成分缓存
        self.wave_components = []
        self._init_wave_field()
        
    def _init_wave_field(self):
        """初始化波浪场 (预计算频率、波数、幅值、相位)"""
        mode = self.p.Env_WaveMod
        
        if mode in [2, 3]: # JONSWAP (2) or PM (3)
            np.random.seed(42) # 保证复现性
            n_freq = 100 # 频率分辨率
            w_min, w_max = 0.1, 3.0
            dw = (w_max - w_min) / n_freq
            omega = np.linspace(w_min, w_max, n_freq)
            
            # PM Spectrum Definition
            Tp = self.p.Env_WavePeriod
            Hs = self.p.Env_WaveHeight
            wp = 2 * np.pi / Tp
            
            # S(w) = 5/16 * Hs^2 * wp^4 * w^-5 * exp(-1.25 * (w/wp)^-4)
            S = (5.0/16.0) * Hs**2 * wp**4 * np.power(omega, -5) * np.exp(-1.25 * np.power(omega/wp, -4))
            
            # JONSWAP Peak Enhancement
            if mode == 2:
                gamma = self.p.Env_WaveGamma
                sigma = np.where(omega <= wp, 0.07, 0.09)
                exponent = -((omega - wp)**2) / (2 * sigma**2 * wp**2)
                r = np.exp(exponent)
                peak_factor = np.power(gamma, r)
                
                # Normalize JONSWAP energy to match Hs
                # 1. Integrate raw JONSWAP
                S_raw = S * peak_factor
                m0_raw = np.sum(S_raw * dw)
                # 2. Target Energy m0 = (Hs/4)^2
                m0_target = (Hs / 4.0)**2
                scale = m0_target / m0_raw
                S = S_raw * scale
            
            # Convert Spectrum to Wave Components (Amplitude = sqrt(2 * S * dw))
            amps = np.sqrt(2 * S * dw)
            phases = np.random.rand(n_freq) * 2 * np.pi
            
            # Dispersion Relation (Deep water approx: k = w^2/g)
            # For finite depth: w^2 = g * k * tanh(k * h) -> solved iteratively if needed
            # Here use Deep Water for stability in V4.1
            k_vals = np.power(omega, 2) / self.g
            
            self.wave_components = list(zip(omega, k_vals, amps, phases))
            print(f"[SeaState] Initialized Irregular Wave Field: {mode} ({n_freq} components)")
            
        elif mode == 1: # Regular (Airy)
            print("[SeaState] Initialized Regular Airy Wave")
        else:
            print("[SeaState] Still Water Initialized")

    def update(self, t):
        """更新内部时间 (供求解器步进调用)"""
        self.time = t

    def get_wave_elevation(self, x, y):
        """
        获取指定水平位置 (x,y) 处的波面高度 eta(t)
        用于计算浮力变化和可视化
        """
        eta = 0.0
        t = self.time
        
        if self.p.Env_WaveMod == 1: # Regular
            H = self.p.Env_WaveHeight
            T = self.p.Env_WavePeriod
            w = 2 * np.pi / (T + 1e-6)
            k = w**2 / self.g
            # Assume wave propagates in +X direction
            eta = 0.5 * H * np.cos(k*x - w*t)
            
        elif self.p.Env_WaveMod in [2, 3]: # Irregular
            for w, k, A, phi in self.wave_components:
                # Superposition
                eta += A * np.cos(k*x - w*t + phi)
                
        return eta

    def get_current_velocity(self, z):
        """
        计算海流剖面 U_current(z)
        OpenFAST Sub-Surface + Near-Surface + Reference Model
        """
        u_curr = self.p.Curr_RefSpeed # Uniform / Reference
        
        # 1. Near-Surface (Linear decay from z=0 to z=-h_ref)
        if self.p.Curr_NearSurfSpeed > 0:
            h_ref = self.p.Curr_RefDepth
            if z >= -h_ref:
                # At z=0 -> U_near; At z=-h_ref -> 0
                factor = (z + h_ref) / h_ref
                u_curr += self.p.Curr_NearSurfSpeed * factor
                
        # 2. Sub-Surface (Power Law from seabed)
        # U(z) = U_sub * ((z + depth) / depth) ^ alpha
        if self.p.Curr_SubSpeed > 0:
            dist_from_bed = z + self.depth
            if dist_from_bed > 0:
                factor = (dist_from_bed / self.depth) ** self.p.Curr_SubExp
                u_curr += self.p.Curr_SubSpeed * factor
                
        # Assume Current is always along X for V4.1
        return np.array([u_curr, 0.0, 0.0])

    def get_kinematics(self, x, y, z):
        """
        核心接口: 获取流体运动学状态
        Returns:
            vel: [u, v, w] (Combined Wave + Current)
            acc: [ax, ay, az] (Wave Acceleration)
            p_dyn: Dynamic Pressure (Bernoulli term approx)
        """
        # 1. Wave Kinematics (Airy / Superposition)
        u_w, v_w, w_w = 0.0, 0.0, 0.0
        ax_w, ay_w, az_w = 0.0, 0.0, 0.0
        
        # Wheeler Stretching implementation (Effective z) to avoid exponential explosion above SWL
        # z_eff = z if z <= 0 else 0
        # More advanced: z_eff = depth * (z + depth) / (depth + eta) - depth
        # For V4.1 stability, use simple cutoff
        z_safe = min(z, 0) 
        
        t = self.time
        
        if self.p.Env_WaveMod == 1: # Regular
            H, T = self.p.Env_WaveHeight, self.p.Env_WavePeriod
            if H > 0:
                w = 2 * np.pi / (T+1e-6)
                k = w**2 / self.g
                decay = np.exp(k * z_safe)
                phase = k*x - w*t
                
                s_ph, c_ph = np.sin(phase), np.cos(phase)
                
                u_w = 0.5 * H * w * decay * c_ph
                w_w = 0.5 * H * w * decay * s_ph
                
                ax_w = 0.5 * H * w**2 * decay * s_ph
                az_w = -0.5 * H * w**2 * decay * c_ph
                
        elif self.p.Env_WaveMod in [2, 3]: # Irregular
            for w, k, A, phi in self.wave_components:
                decay = np.exp(k * z_safe)
                phase = k*x - w*t + phi
                s_ph, c_ph = np.sin(phase), np.cos(phase)
                
                u_w += A * w * decay * c_ph
                w_w += A * w * decay * s_ph
                
                ax_w += A * w**2 * decay * s_ph
                az_w -= A * w**2 * decay * c_ph
                
        # 2. Current Kinematics
        vel_curr = self.get_current_velocity(z)
        
        # 3. Total Kinematics
        vel_total = np.array([u_w, 0.0, w_w]) + vel_curr # Assume wave along X
        acc_total = np.array([ax_w, 0.0, az_w]) # Current acceleration is 0 (steady)
        
        # 4. Dynamic Pressure (Linearized: rho * g * eta * decay)
        # Or Bernoulli: -rho * dphi/dt
        # Simple approx for PotFlow excitation:
        p_dyn = self.rho * self.g * (u_w / (self.g/np.pi)) # Rough scaling, used only for monitoring
        
        return vel_total, acc_total, p_dyn