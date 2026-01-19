from PySide6.QtCore import QThread, Signal
import time
import numpy as np
import traceback
from collections import deque

from physics.aerodynamics import Aerodynamics
from physics.hydrodynamics import HydroDynamics
from physics.mooring import MooringSystem
from physics.structure_kane_v1 import KaneStructureModel
from physics.inflow import InflowManager
from physics.seastate import SeaState
from control.controller_interface import ControllerInterface
from solver.state_adapter import StateAdapter, AdapterConfig


class SimulationWorker(QThread):
    data_signal = Signal(dict)
    progress_signal = Signal(float, float)
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, params, t_total, dt, ui_fps=30, out_fps=20):
        """
        ui_fps : UI 刷新最大帧率（降低卡顿）
        out_fps: 输出/日志写入的最大帧率（对标 OpenFAST 的 OutDec/采样思想）
        """
        super().__init__()
        self.params = params
        self.t_total = float(t_total)
        self.dt = float(dt)

        self.running = True
        self.paused = False
        self.version = "0.0.1"

        # ---------------------------
        # 节流：计算高频，UI低频
        # ---------------------------
        self.ui_fps = int(max(1, ui_fps))
        self.ui_stride = max(1, int(round(1.0 / (self.ui_fps * self.dt))))

        # 日志/输出节流（你可以让 CSV 更接近 OpenFAST 的 OutDec）
        self.out_fps = int(max(1, out_fps))
        self.out_stride = max(1, int(round(1.0 / (self.out_fps * self.dt))))

        # cache
        self.log_cache = {}

    def run(self):
        try:
            dt = self.dt
            t = 0.0

            # Init Physics Modules
            inflow = InflowManager(self.params)
            seastate = SeaState(self.params)

            aero = Aerodynamics(self.params)
            hydro = HydroDynamics(self.params, seastate)
            mooring = MooringSystem(self.params, seastate, dt)
            structure = KaneStructureModel(self.params)
            controller = ControllerInterface(dt)

            # 状态向量
            state = np.zeros(44)
            state[2] = 0.0
            state[33] = 6.0 * 2 * np.pi / 60.0

            history_dydt = deque(maxlen=4)

            # DOF Names matching Control Dock
            dof_names = [
                "Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw",
                "TwrFA1", "TwrSS1", "TwrFA2", "TwrSS2",
                "NacYaw", "GenAz", "DrTr",
                "B1F1", "B1E1", "B1F2",
                "B2F1", "B2E1", "B2F2",
                "B3F1", "B3E1", "B3F2"
            ]

            print(f"Simulation V{self.version} started. Solver: Adams-Bashforth-Moulton 4.")
            adapter = StateAdapter(AdapterConfig(angles_are_rad=True))
            wall_start = time.time()

            # 为避免闭包里引用外层 t 造成混乱：用 step_count 判断“主步”
            step_count = 0

            def compute_forces_and_log(curr_t, curr_state, do_log=False):
                """
                do_log=True 时，更新 self.log_cache（节流到 out_stride）
                """
                # 1. Env
                inflow.update(curr_t)
                seastate.update(curr_t)
                wind_vec = inflow.get_wind_at_point(curr_t, 0, 0, self.params.HubHeight)
                v_wind = float(np.linalg.norm(wind_vec))

                # 2. Control
                pitch_deg = float(np.degrees(curr_state[4]))
                gen_torq, pitch_cmd = controller.update(curr_t, curr_state[33], pitch_deg, v_wind)
                gen_torq = float(gen_torq)
                pitch_cmd = float(pitch_cmd)

                # 3. Rigid state (平台6位姿 + 6速度)
                rigid_state = np.zeros(12)
                rigid_state[:6] = curr_state[:6]
                rigid_state[6:] = curr_state[22:28]

                # 4. Loads
                f_gen_aero = aero.update(curr_state, curr_t, v_wind, pitch_cmd)  # 22
                f_hydro = hydro.update(rigid_state, curr_t)                      # 14 or >=6
                f_moor = mooring.update(rigid_state, curr_t)                     # 14 or >=6

                # 汇总到结构广义力（22）
                F_gen_total = np.zeros(22)
                F_gen_total += f_gen_aero
                F_gen_total[:6] += f_hydro[:6] + f_moor[:6]
                F_gen_total[11] -= gen_torq
                F_gen_total[12] += gen_torq

                # 5. Dynamics
                dydt = structure.get_derivative(curr_state, F_gen_total, rotor_speed=curr_state[33])

                # 6. Logging (节流输出，更像 OpenFAST OutDec)
                if do_log:
                    wave_elev = float(seastate.get_wave_elevation(0, 0))
                    hss_spd = float(curr_state[33] * self.params.GearboxRatio)
                    pwr = float(gen_torq * hss_spd * self.params.GenEfficiency)

                    # 平台位移/速度（OpenFAST 风格命名，便于对标）
                    # 你 state[22:28] 在你原逻辑里就是平台6速度块
                    self.log_cache = {
                        # ----- Time -----
                        "Time": float(curr_t),

                        # ----- Environment -----
                        "WindSpeed": v_wind,
                        "WaveElev": wave_elev,

                        # ----- Control / drivetrain -----
                        "RotSpeed": float(curr_state[33]),        # rad/s
                        "HSS_Spd": hss_spd,                       # rad/s
                        "GenTq": gen_torq,                        # N·m
                        "GenPwr": pwr,                            # W
                        "PitchCmd": pitch_cmd,                    # deg or rad? 你 controller 输出一般是 deg（保持你原记录）
                        "AeroThrust": float(getattr(aero, "last_thrust", 0.0)),

                        # ----- Platform motions -----
                        "PtfmSurge": float(curr_state[0]),
                        "PtfmSway":  float(curr_state[1]),
                        "PtfmHeave": float(curr_state[2]),
                        "PtfmRoll":  float(curr_state[3]),
                        "PtfmPitch": float(curr_state[4]),
                        "PtfmYaw":   float(curr_state[5]),

                        "PtfmSurgeVel": float(curr_state[22]),
                        "PtfmSwayVel":  float(curr_state[23]),
                        "PtfmHeaveVel": float(curr_state[24]),
                        "PtfmRollVel":  float(curr_state[25]),
                        "PtfmPitchVel": float(curr_state[26]),
                        "PtfmYawVel":   float(curr_state[27]),

                        # ----- Loads split (6 components) -----
                        "HydroFx": float(f_hydro[0]), "HydroFy": float(f_hydro[1]), "HydroFz": float(f_hydro[2]),
                        "HydroMx": float(f_hydro[3]), "HydroMy": float(f_hydro[4]), "HydroMz": float(f_hydro[5]),

                        "MoorFx": float(f_moor[0]), "MoorFy": float(f_moor[1]), "MoorFz": float(f_moor[2]),
                        "MoorMx": float(f_moor[3]), "MoorMy": float(f_moor[4]), "MoorMz": float(f_moor[5]),

                        # aero 的平台6分量（如果你的 aero.update 前6就是平台气动力/矩，就能直接对标）
                        "AeroFx": float(f_gen_aero[0]), "AeroFy": float(f_gen_aero[1]), "AeroFz": float(f_gen_aero[2]),
                        "AeroMx": float(f_gen_aero[3]), "AeroMy": float(f_gen_aero[4]), "AeroMz": float(f_gen_aero[5]),
                    }

                return dydt

            while self.running and t < self.t_total:
                if self.paused:
                    time.sleep(0.1)
                    continue

                step_start = time.time()

                # --- 主步：计算 f_n 并（按 out_stride）更新 log_cache ---
                do_log = (step_count % self.out_stride == 0)
                f_n = compute_forces_and_log(t, state, do_log=do_log)
                history_dydt.append(f_n)

                # --- 积分 ---
                if step_count < 3:  # RK4 起步
                    k1 = f_n
                    k2 = compute_forces_and_log(t + dt/2, state + dt/2 * k1, do_log=False)
                    k3 = compute_forces_and_log(t + dt/2, state + dt/2 * k2, do_log=False)
                    k4 = compute_forces_and_log(t + dt,   state + dt * k3,   do_log=False)
                    state_next = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
                else:  # ABM4
                    f0, f1, f2, f3 = history_dydt[-1], history_dydt[-2], history_dydt[-3], history_dydt[-4]
                    state_pred = state + (dt/24.0) * (55*f0 - 59*f1 + 37*f2 - 9*f3)
                    f_pred = compute_forces_and_log(t + dt, state_pred, do_log=False)
                    state_next = state + (dt/24.0) * (9*f_pred + 19*f0 - 5*f1 + f2)

                state = state_next
                t += dt
                step_count += 1

                # --- 构造 Data 包（注意：UI 限帧输出） ---
                if step_count % self.ui_stride == 0:
                    data = {'time': float(t)}

                    # DOF（你原有通道保持不变）
                    for i, name in enumerate(dof_names):
                        data[f'dof_{name}'] = float(state[i])

                    # 扩展输出（OpenFAST-like）
                    data.update(self.log_cache)

                    # 适配到可视化接口 q*/env_*（保留你的机制）
                    data = adapter.adapt(data)

                    self.data_signal.emit(data)

                # 进度条也可以节流一下（避免 UI 主线程太忙）
                if step_count % self.ui_stride == 0:
                    self.progress_signal.emit((t / self.t_total) * 100.0, time.time() - wall_start)

                # 轻微让出时间片（可选）
                if time.time() - step_start < 0.001:
                    time.sleep(0.0001)

        except Exception:
            self.error_signal.emit(traceback.format_exc())

        self.finished_signal.emit()

    def stop(self):
        self.running = False
