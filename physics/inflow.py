import numpy as np
import os
import struct

class BTSReader:
    """
    [V4.4.1 Fixed] Reader for TurSim .bts binary files (ID=7/8)
    Reference: TurbSim User's Guide Appendix D (Table D-1 & D-2)
    Fix: Added missing 'n_tower' field which caused alignment issues.
    """
    def __init__(self, filename):
        self.filename = filename
        self.ok = False
        self.data = None
        self.duration = 1.0 
        self.dt = 0.1
        self.nt = 0
        self.load()

    def load(self):
        if not os.path.exists(self.filename):
            print(f"[BTS Error] File not found: {self.filename}")
            return

        try:
            with open(self.filename, 'rb') as f:
                # --- 1. Header Parsing (Table D-1) ---
                
                # ID: Integer (2)
                self.id = struct.unpack('<h', f.read(2))[0]
                if self.id not in [7, 8]:
                    print(f"[BTS Error] Unsupported ID: {self.id} (Expected 7 or 8)")
                    return

                # NumGrid_Z: Integer (4)
                self.nz = struct.unpack('<i', f.read(4))[0]
                
                # NumGrid_Y: Integer (4)
                self.ny = struct.unpack('<i', f.read(4))[0]
                
                # n_tower: Integer (4) <-- [CRITICAL FIX] 之前版本漏读了这一项
                self.n_tower = struct.unpack('<i', f.read(4))[0]
                
                # nt: Integer (4) [现在指针位置正确了]
                self.nt = struct.unpack('<i', f.read(4))[0]
                
                # Grid Resolutions: Real (4) x 3
                self.dz = struct.unpack('<f', f.read(4))[0]
                self.dy = struct.unpack('<f', f.read(4))[0]
                self.dt = struct.unpack('<f', f.read(4))[0]
                
                # Reference: Real (4) x 3
                self.uhub = struct.unpack('<f', f.read(4))[0]
                self.hubht = struct.unpack('<f', f.read(4))[0]
                self.zbottom = struct.unpack('<f', f.read(4))[0]
                
                # Scaling & Offset: Real (4) x 6
                self.scl = np.array(struct.unpack('<fff', f.read(12)))
                self.off = np.array(struct.unpack('<fff', f.read(12)))
                
                # Description
                self.nchar = struct.unpack('<i', f.read(4))[0]
                if self.nchar > 0:
                    f.read(self.nchar) # Skip string
                
                # --- 2. Validation ---
                if self.nt <= 0:
                    print(f"[BTS Error] Invalid time steps nt={self.nt}. Check file integrity.")
                    return
                
                self.duration = self.nt * self.dt
                
                # --- 3. Data Reading (Table D-2) ---
                # Data Layout: Time (outer) -> Z -> Y -> Component (inner)
                # Block size per time step = (GridPoints + TowerPoints) * 3
                
                n_grid_values = self.nz * self.ny * 3
                total_values_per_step = n_grid_values + (self.n_tower * 3)
                total_file_values = self.nt * total_values_per_step
                
                # Read all data as Int16
                raw_data = np.fromfile(f, dtype=np.int16)
                
                # Robustness check for truncated files
                if raw_data.size < total_file_values:
                    # If very close, might be footer missing, try to proceed
                    if raw_data.size < total_file_values * 0.99: 
                        print(f"[BTS Error] File truncated. Expected {total_file_values}, got {raw_data.size}")
                        return
                    # Trim to fit what we have (or pad?) -> Trim safest
                    available_steps = raw_data.size // total_values_per_step
                    self.nt = available_steps
                    raw_data = raw_data[:self.nt * total_values_per_step]
                    self.duration = self.nt * self.dt
                    print(f"[BTS Warning] File truncated, loaded {self.nt} steps.")

                # Reshape to separate time steps
                raw_steps = raw_data.reshape((self.nt, total_values_per_step))
                
                # Extract Grid Data (Discard tower data at end of each step)
                grid_raw = raw_steps[:, :n_grid_values]
                
                # Reshape Grid Data
                # Official Spec D-2 Loop Order: 
                # 1. it (Time)
                # 2. iz (Vertical Z)
                # 3. iy (Horizontal Y)
                # 4. i (Component)
                # So shape is (nt, nz, ny, 3)
                self.data_int = grid_raw.reshape((self.nt, self.nz, self.ny, 3))
                
                # Apply Scaling (Table D-2 Eq D-1):
                # V = (V_norm - V_intercept) / V_slope
                # Note: Code usually implements V = (Int - Off) / Scl based on OpenFAST convention.
                safe_scl = np.where(self.scl == 0, 1.0, self.scl)
                self.data = (self.data_int - self.off) / safe_scl
                
                # Coordinate Vectors
                self.z_grid = self.zbottom + np.arange(self.nz) * self.dz
                width = (self.ny - 1) * self.dy
                self.y_grid = np.linspace(-width/2, width/2, self.ny)
                
                self.ok = True
                print(f"[Inflow] BTS Loaded: {self.nt} steps, {self.duration:.1f}s, Hub={self.uhub}m/s")

        except Exception as e:
            print(f"[BTS Error] Load failed: {e}")
            self.ok = False

class InflowManager:
    """
    WOUSE V4.4.1 Inflow Module
    """
    def __init__(self, params):
        self.params = params
        self.time = 0.0
        self._turb_components = []
        self.bts = None
        self._init_wind_source()

    def _init_wind_source(self):
        if self.params.Env_WindType == 1: # Internal
            self._init_internal_turbulence()
            
        elif self.params.Env_WindType == 2: # External
            if self.params.Env_WindFile and os.path.exists(self.params.Env_WindFile):
                print(f"[Inflow] Loading: {self.params.Env_WindFile}")
                self.bts = BTSReader(self.params.Env_WindFile)
                if not self.bts.ok:
                    print("[Warning] BTS failed. Reverting to Steady.")
                    self.params.Env_WindType = 0 
            else:
                print("[Warning] No file. Reverting to Steady.")
                self.params.Env_WindType = 0 

    def _init_internal_turbulence(self):
        np.random.seed(42) 
        num_components = 50
        freqs = np.logspace(np.log10(0.01), np.log10(2.0), num_components)
        phases = np.random.rand(num_components) * 2 * np.pi
        target_sigma = (self.params.Env_TurbIntensity / 100.0) * self.params.Env_WindSpeed
        amplitudes = np.ones(num_components) 
        scale = target_sigma / np.sqrt(np.sum(amplitudes**2) / 2)
        amplitudes *= scale
        self._turb_components = list(zip(freqs, amplitudes, phases))

    def get_wind_at_point(self, t, x, y, z):
        # 1. Steady / Internal
        if self.params.Env_WindType == 0 or self.params.Env_WindType == 1:
            z_eff = max(z, 1.0)
            shear = (z_eff / self.params.HubHeight) ** self.params.Env_ShearExp
            u_base = self.params.Env_WindSpeed * shear
            u_turb = 0.0
            if self.params.Env_WindType == 1:
                t_eff = t - (x / max(self.params.Env_WindSpeed, 1.0))
                for f, A, phi in self._turb_components:
                    u_turb += A * np.sin(2 * np.pi * f * t_eff + phi)
            return np.array([u_base + u_turb, 0.0, 0.0])

        # 2. External (BTS)
        # Check duration > 0 to be absolutely safe against ZeroDivision
        elif self.params.Env_WindType == 2 and self.bts and self.bts.ok and self.bts.duration > 0.001:
            # Taylor's Hypothesis: Advection
            # Use GUI wind speed to control advection speed if desired, or file hub speed
            u_adv = max(self.params.Env_WindSpeed, 0.1) 
            t_lag = x / u_adv
            t_lookup = t - t_lag
            
            # Cyclic Wrap
            t_lookup = t_lookup % self.bts.duration 
            
            # Indexing
            idx_t = int(t_lookup / self.bts.dt)
            idx_t = min(idx_t, self.bts.nt - 1)
            
            idx_z = int((z - self.bts.zbottom) / self.bts.dz)
            idx_z = np.clip(idx_z, 0, self.bts.nz - 1)
            
            y_min = self.bts.y_grid[0]
            idx_y = int((y - y_min) / self.bts.dy)
            idx_y = np.clip(idx_y, 0, self.bts.ny - 1)
            
            # Query format [t, z, y] matches self.data shape [nt, nz, ny, 3]
            return self.bts.data[idx_t, idx_z, idx_y, :]

        return np.array([self.params.Env_WindSpeed, 0.0, 0.0])

    def update(self, t):
        self.time = t