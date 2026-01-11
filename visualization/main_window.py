from PySide6.QtWidgets import (QMainWindow, QDockWidget, QMessageBox)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction

from config.fast_params import SystemParams
from solver.worker import SimulationWorker
from visualization.widgets.dock_control import ControlDockWidget
from visualization.widgets.dock_3d import View3DDockWidget
from visualization.widgets.dock_scope import ScopeDockWidget
from visualization.dialog_env import EnvironmentDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"LabFAST V0.0.1 (AeroDyn 15 + Kane 22-DOF)")
        self.resize(1600, 1000)

        self.params = SystemParams()
        self.worker = None

        self.init_docks()
        self.init_menu()

    def init_docks(self):
        # 1. Control Panel
        self.dock_ctrl = QDockWidget("Control Panel", self)
        self.dock_ctrl.setObjectName("DockCtrl")
        self.widget_ctrl = ControlDockWidget()
        self.dock_ctrl.setWidget(self.widget_ctrl)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_ctrl)

        # 2. Oscilloscope
        self.dock_scope = QDockWidget("Oscilloscope & Data", self)
        self.dock_scope.setObjectName("DockScope")
        self.widget_scope = ScopeDockWidget()
        self.dock_scope.setWidget(self.widget_scope)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_scope)

        # 3. 3D View
        self.dock_3d = QDockWidget("3D Digital Twin", self)
        self.dock_3d.setObjectName("Dock3D")
        self.widget_3d = View3DDockWidget(self.params)
        self.dock_3d.setWidget(self.widget_3d)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_3d)

        # --- Signal Connections ---
        self.widget_ctrl.sig_start_sim.connect(self.start_simulation)
        self.widget_ctrl.sig_stop_sim.connect(self.stop_simulation)
        self.widget_ctrl.sig_open_env.connect(self.open_env_dialog)

        # Control -> Scope (Channel Selection)
        self.widget_ctrl.sig_channels_changed.connect(self.widget_scope.update_channels)

        # 3D View -> Main (Time Travel / Replay)
        self.widget_3d.sig_frame_request.connect(self.on_3d_frame_request)

    def init_menu(self):
        menu = self.menuBar()

        view_menu = menu.addMenu("View")
        view_menu.addAction(self.dock_ctrl.toggleViewAction())
        view_menu.addAction(self.dock_scope.toggleViewAction())
        view_menu.addAction(self.dock_3d.toggleViewAction())

        sett_menu = menu.addMenu("Settings")
        sett_menu.addAction("Environment...", self.open_env_dialog)

    def open_env_dialog(self):
        dlg = EnvironmentDialog(self.params, self)
        if dlg.exec():
            self.params = dlg.get_params()

    def start_simulation(self, t_total, dt):
        self.widget_ctrl.btn_run.setEnabled(False)
        self.widget_ctrl.btn_stop.setEnabled(True)
        self.widget_scope.reset_session()

        self.worker = SimulationWorker(self.params, t_total, dt)
        self.worker.data_signal.connect(self.on_data_update)
        self.worker.progress_signal.connect(self.widget_ctrl.update_progress)
        self.worker.finished_signal.connect(self.on_sim_finished)
        self.worker.error_signal.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.start()

    def stop_simulation(self):
        if self.worker:
            self.worker.stop()

    @Slot(dict)
    def on_data_update(self, data):
        # 1. Update Scope
        if self.dock_scope.isVisible():
            self.widget_scope.update_data(data)

        # 2. Update 3D View
        if self.dock_3d.isVisible():
            if hasattr(self.widget_scope, 'data_history'):
                total_frames = len(self.widget_scope.data_history)
            else:
                total_frames = 0
            self.widget_3d.update_timeline(total_frames, data['time'])

            if self.widget_3d.chk_sync.isChecked():
                self._update_3d_view(data)

        # 3. Update Power Label (Use updated key: env_GenPower)
        p_mw = data.get('env_GenPower', 0.0) / 1e6
        self.widget_ctrl.lbl_power.setText(f"Power: {p_mw:.2f} MW")

    def _update_3d_view(self, data):
        """Map new dof_ keys to 3D view"""
        state_rigid = [
            data.get('dof_Surge', 0),
            data.get('dof_Sway', 0),
            data.get('dof_Heave', 0),
            data.get('dof_Roll', 0),
            data.get('dof_Pitch', 0),
            data.get('dof_Yaw', 0)
        ]
        self.widget_3d.update_pose(state_rigid, data['time'])

    @Slot(int)
    def on_3d_frame_request(self, idx):
        history = self.widget_scope.data_history
        if 0 <= idx < len(history):
            data = history[idx]
            self._update_3d_view(data)
            self.widget_3d.lbl_time.setText(f"T: {data['time']:.2f}s")

    @Slot()
    def on_sim_finished(self):
        self.widget_ctrl.btn_run.setEnabled(True)
        self.widget_ctrl.btn_stop.setEnabled(False)
        self.widget_ctrl.progress.setValue(100)

        log_path = getattr(self.widget_scope, 'current_log_path', 'Unknown')
        QMessageBox.information(self, "Simulation Finished",
                                f"Simulation Completed.\nData saved to:\n{log_path}")