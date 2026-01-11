import numpy as np
from scipy.optimize import root_scalar

# --- 1. 原有的 Simple Solver (重命名为 InternalSimpleMooring) ---
class CatenaryLine:
    """单根悬链线求解器 (V4.1.1) - 保持原逻辑不变"""
    def __init__(self, params, anchor_pos, fairlead_local_pos):
        self.p = params
        self.anchor = np.array(anchor_pos)
        self.fairlead_local = np.array(fairlead_local_pos)
        self.L = params.Moor_LineLength
        self.w = params.Moor_LineMass * params.Gravity 
        self.EA = params.Moor_LineEA

    def solve_force(self, fairlead_global):
        delta = fairlead_global - self.anchor
        X_target = np.sqrt(delta[0]**2 + delta[1]**2)
        if X_target < 1.0: X_target = 1.0
        Z_target = delta[2]

        def span_mismatch(H):
            if H <= 1.0: H = 1.0
            term = 2.0 * H / self.w
            val = Z_target * (Z_target + term)
            if val < 0: val = 0
            L_s = np.sqrt(val)
            if L_s >= self.L: L_s = self.L
            x_s = (H / self.w) * np.arcsinh(self.w * L_s / H)
            x_b = self.L - L_s
            estretch = (H * self.L) / self.EA
            return (x_s + x_b + estretch) - X_target

        try:
            sol = root_scalar(span_mismatch, bracket=[100, 1e8], method='brentq', xtol=0.1)
            H = sol.root
        except: H = 1e5 
            
        term = 2.0 * H / self.w
        L_s = np.sqrt(Z_target * (Z_target + term))
        if L_s > self.L: L_s = self.L
        V = self.w * L_s
        
        dir_vec = delta / np.linalg.norm(delta)
        dir_horiz = np.array([dir_vec[0], dir_vec[1], 0])
        if np.linalg.norm(dir_horiz) > 1e-6:
            dir_horiz = dir_horiz / np.linalg.norm(dir_horiz)
        return -H * dir_horiz + np.array([0, 0, -V])

class InternalSimpleMooring:
    """原 MooringSystem 逻辑"""
    def __init__(self, params, seastate, dt=None):
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

    def _setup_lines(self):
        self._load_custom_points()
        n = int(self.fairlead_points.shape[0])
        for i in range(n):
            fl = self.fairlead_points[i]
            an = self.anchor_points[i]
            self.lines.append(CatenaryLine(self.p, an, fl))

    def update(self, state, t):
        surge, sway, heave = state[0], state[1], state[2]
        roll, pitch, yaw = state[3], state[4], state[5]
        plat_pos = np.array([surge, sway, heave])
        R = np.eye(3) 
        R[0,1] = -yaw; R[0,2] = pitch
        R[1,0] = yaw;  R[1,2] = -roll
        R[2,0] = -pitch; R[2,1] = roll
        
        F_total = np.zeros(6)
        v_curr_avg = self.seastate.get_current_velocity(-100.0)
        
        for line in self.lines:
            fl_global = plat_pos + R @ line.fairlead_local
            f_vec = line.solve_force(fl_global)
            
            # Drag Force (Only in simple model)
            v_rel = v_curr_avg - np.array([state[6], state[7], state[8]])
            f_drag = 0.5 * 1025.0 * self.p.Moor_DragCoeff * 0.2 * line.L * v_rel * np.linalg.norm(v_rel)
            f_drag[2] *= 0.1 
            
            f_line_total = f_vec + f_drag
            F_total[0:3] += f_line_total
            r_vec = fl_global - plat_pos
            F_total[3:6] += np.cross(r_vec, f_line_total)
            
        F_ext = np.zeros(14)
        F_ext[0:6] = F_total
        return F_ext
    
    def close(self):
        pass
    
    
# --- 2. 主入口类 (MooringSystem) ---
class MooringSystem:
    """
    统一接口类，根据配置分发到具体的求解器
    """
    def __init__(self, params, seastate, dt):
        self.model_type = params.Moor_ModelType
        self.solver = None
        
        print(f"Initializing Mooring System. Type: {self.model_type}")
        
        if self.model_type == 0:
            # 0: Simple Internal Solver
            self.solver = InternalSimpleMooring(params, seastate, dt)
            
        elif self.model_type == 1:
            # 1: Catenary Moor 
            try:
                from .catenary_moor import CatenaryMoor
                self.solver = CatenaryMoor(params, seastate)
            except ImportError:
                print("Warning: CatenaryMoor module not found. Falling back to Simple.")
                self.solver = InternalSimpleMooring(params, seastate, dt)
                
        elif self.model_type == 2:
            # 2: MoorDyn (External)
            try:
                from .moordyn.solver import MoorDynWrapper
                self.solver = MoorDynWrapper(params, seastate, dt)
            except Exception as e:
                print(f"Error loading MoorDyn: {e}. Falling back to Simple.")
                self.solver = InternalSimpleMooring(params, seastate, dt)
                
        elif self.model_type == 3:
            # 3: MoorEmm (Python Native Lumped Mass)
            try:
                # [Modified] 注意这里引用的文件名是 moor_emm_solver
                from .moor_emm.moor_emm_solver import MoorEmmSolver
                self.solver = MoorEmmSolver(params, seastate, dt)
            except Exception as e:
                print(f"Error loading MoorEmm: {e}. Falling back to Simple.")
                self.solver = InternalSimpleMooring(params, seastate, dt)
        
        else:
            self.solver = InternalSimpleMooring(params, seastate, dt)

    def update(self, state, t):
        return self.solver.update(state, t)

    def close(self):
        if hasattr(self.solver, 'close'):
            self.solver.close()
