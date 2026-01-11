from PySide6.QtCore import QThread, Signal
import time
import numpy as np
import traceback
from collections import deque

from physics.aerodynamics import Aerodynamics
from physics.hydrodynamics import HydroDynamics
from physics.moor_emm_solver import MoorEmmSolver
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

    def __init__(self, params, t_total, dt):
        super().__init__()
        self.params = params
        self.t_total = t_total
        self.dt = dt
        self.running = True
        self.paused = False
        self.version = "0.0.1"

    def run(self):
        try:
            dt = self.dt
            t = 0.0

            # Init Physics Modules
            inflow = InflowManager(self.params)
            seastate = SeaState(self.params)

            aero = Aerodynamics(self.params)
            hydro = HydroDynamics(self.params, seastate)
            mooring = MoorEmmSolver(self.params, seastate, dt)
            structure = KaneStructureModel(self.params)
            controller = ControllerInterface(dt)

            state = np.zeros(44)
            state[2] = 0.0
            state[33] = 6.0 * 2 * np.pi / 60.0

            history_dydt = deque(maxlen=4)
            self.log_cache = {}

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

            def get_system_derivative(curr_t, curr_state):
                # 1. Env
                inflow.update(curr_t)
                seastate.update(curr_t)
                wind_vec = inflow.get_wind_at_point(curr_t, 0, 0, self.params.HubHeight)
                v_wind = np.linalg.norm(wind_vec)

                # 2. Control
                pitch_deg = np.degrees(curr_state[4])
                gen_torq, pitch_cmd = controller.update(curr_t, curr_state[33], pitch_deg, v_wind)

                # 3. Forces
                rigid_state = np.zeros(12)
                rigid_state[:6] = curr_state[:6]; rigid_state[6:] = curr_state[22:28]

                f_gen_aero = aero.update(curr_state, curr_t, v_wind, pitch_cmd)
                f_hydro = hydro.update(rigid_state, curr_t)
                f_moor = mooring.update(rigid_state, curr_t)

                F_gen_total = np.zeros(22)
                F_gen_total += f_gen_aero
                F_gen_total[:6] += f_hydro[:6] + f_moor[:6]
                F_gen_total[11] -= gen_torq
                F_gen_total[12] += gen_torq

                # 4. Dynamics
                dydt = structure.get_derivative(curr_state, F_gen_total, rotor_speed=curr_state[33])

                # 5. Logging (Main Step Only)
                if abs(curr_t - t) < 1e-6:
                    wave_elev = seastate.get_wave_elevation(0, 0)
                    hss_spd = curr_state[33] * self.params.GearboxRatio
                    pwr = gen_torq * hss_spd * self.params.GenEfficiency

                    self.log_cache = {
                        # --- Environment & Output (env_) ---
                        # [Modified] All outputs get 'env_' prefix to be saved in envdata.csv
                        'env_Time': curr_t,
                        'env_WindSpeed': v_wind,
                        'env_WaveElev': wave_elev,
                        'env_GenTorque': gen_torq,
                        'env_GenPower': pwr,
                        'env_AeroThrust': aero.last_thrust,
                        'env_PitchCmd': pitch_cmd,

                        # --- Mooring (moor_) ---
                        'moor_Time': curr_t,
                        'moor_Fx': f_moor[0], 'moor_Fy': f_moor[1], 'moor_Fz': f_moor[2],
                        'moor_Mx': f_moor[3], 'moor_My': f_moor[4], 'moor_Mz': f_moor[5]
                    }

                return dydt

            step_count = 0
            while self.running and t < self.t_total:
                if self.paused: time.sleep(0.1); continue

                step_start = time.time()

                f_n = get_system_derivative(t, state)
                history_dydt.append(f_n)

                if step_count < 3: # RK4
                    k1 = f_n
                    k2 = get_system_derivative(t + dt/2, state + dt/2 * k1)
                    k3 = get_system_derivative(t + dt/2, state + dt/2 * k2)
                    k4 = get_system_derivative(t + dt,   state + dt * k3)
                    state_next = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
                else: # ABM4
                    f0, f1, f2, f3 = history_dydt[-1], history_dydt[-2], history_dydt[-3], history_dydt[-4]
                    state_pred = state + (dt/24.0) * (55*f0 - 59*f1 + 37*f2 - 9*f3)
                    f_pred = get_system_derivative(t + dt, state_pred)
                    state_next = state + (dt/24.0) * (9*f_pred + 19*f0 - 5*f1 + f2)

                state = state_next
                t += dt
                step_count += 1

                # 在 while 循环外（更好）或 run() 前面初始化一次：
                # adapter = StateAdapter(AdapterConfig(angles_are_rad=True))

                # --- Construct Data Package ---
                data = {'time': t}

                for i, name in enumerate(dof_names):
                    data[f'dof_{name}'] = state[i]

                data.update(self.log_cache)

                # ★关键：强制替换接口（适配到可视化通道 q*/env_*）
                data = adapter.adapt(data)

                self.data_signal.emit(data)

                self.progress_signal.emit((t/self.t_total)*100, time.time() - wall_start)

                if time.time() - step_start < 0.001: time.sleep(0.0001)

        except Exception as e:
            self.error_signal.emit(traceback.format_exc())

        self.finished_signal.emit()

    def stop(self):
        self.running = False