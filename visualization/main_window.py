from PySide6.QtWidgets import (QMainWindow, QMessageBox, QWidget, QVBoxLayout,
                               QHBoxLayout, QFrame, QPushButton, QFileDialog)
from PySide6.QtCore import Slot

from config.fast_params import SystemParams
from solver.worker import SimulationWorker
from visualization.widgets.dock_control import ControlDockWidget
from visualization.widgets.dock_3d import View3DDockWidget
from visualization.widgets.dock_scope import ScopeDockWidget
from visualization.dialog_env import EnvironmentDialog
from visualization.edit_dialogs import WindDialog, WaveCurrentDialog, MooringDialog, StructureDialog
from visualization.widgets.live_data_panel import LiveDataPanel
from visualization.widgets.observer_panel import ObserverPanel
from solver.data_logger import LogConfig


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"LabFAST V0.0.1 (AeroDyn 15 + Kane 22-DOF)")
        self.resize(1600, 1000)

        self.params = SystemParams()
        self.worker = None

        self.data_history = []
        self.init_layout()
        self.init_menu()

    def init_layout(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        self.widget_3d = View3DDockWidget(self.params)
        self.widget_3d.sig_frame_request.connect(self.on_3d_frame_request)
        self.widget_3d.sig_edit_wind.connect(self.open_wind_dialog)
        self.widget_3d.sig_edit_wave.connect(self.open_wave_dialog)
        self.widget_3d.sig_edit_mooring.connect(self.open_mooring_dialog)
        self.widget_3d.sig_edit_structure.connect(self.open_structure_dialog)

        main_layout.addWidget(self.widget_3d, stretch=7)

        bottom_bar = self._build_bottom_bar()
        main_layout.addWidget(bottom_bar, stretch=3)

        self.setCentralWidget(central)

    def _build_bottom_bar(self):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setSpacing(16)
        layout.setContentsMargins(0, 8, 0, 0)

        self.widget_ctrl = ControlDockWidget()
        self.widget_scope = ScopeDockWidget()
        self.live_panel = LiveDataPanel()

        operator_panel = self._wrap_cyber_panel([self.widget_ctrl, self.live_panel, self.widget_scope])
        menu_panel = self._build_menu_panel()
        self.observer_panel = ObserverPanel()
        observer_panel = self._wrap_cyber_panel([self.observer_panel])

        layout.addWidget(operator_panel, stretch=4)
        layout.addWidget(menu_panel, stretch=2)
        layout.addWidget(observer_panel, stretch=4)

        self.widget_ctrl.sig_start_sim.connect(self.start_simulation)
        self.widget_ctrl.sig_stop_sim.connect(self.stop_simulation)
        self.widget_ctrl.sig_open_env.connect(self.open_env_dialog)
        self.widget_ctrl.sig_channels_changed.connect(self.widget_scope.update_channels)

        return container

    def _wrap_cyber_panel(self, widgets):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            "QFrame {"
            "background-color: #0b0f1a;"
            "border: 1px solid #19d3ff;"
            "border-radius: 8px;"
            "}"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        for widget in widgets:
            layout.addWidget(widget)
        return frame

    def _build_menu_panel(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            "QFrame {"
            "background-color: #070b14;"
            "border: 1px solid #7cf4ff;"
            "border-radius: 10px;"
            "}"
            "QPushButton {"
            "color: #d7faff;"
            "background-color: #0f1c2c;"
            "border: 1px solid #19d3ff;"
            "border-radius: 6px;"
            "padding: 10px 14px;"
            "font-weight: 600;"
            "}"
            "QPushButton:hover {"
            "background-color: #12304a;"
            "}"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addStretch()
        self.btn_save_file = QPushButton("保存文件")
        self.btn_power = QPushButton("实施功率: 0.00 MW")
        self.btn_runtime = QPushButton("运行时间: 0.00 s")
        self.btn_save_file.clicked.connect(self.on_save_file)
        layout.addWidget(self.btn_save_file)
        layout.addWidget(self.btn_power)
        layout.addWidget(self.btn_runtime)
        layout.addStretch()
        return frame

    def init_menu(self):
        menu = self.menuBar()

        sett_menu = menu.addMenu("Settings")
        sett_menu.addAction("Environment...", self.open_env_dialog)
        sett_menu.addAction("Wind...", self.open_wind_dialog)
        sett_menu.addAction("Wave/Current...", self.open_wave_dialog)
        sett_menu.addAction("Mooring...", self.open_mooring_dialog)
        sett_menu.addAction("Structure...", self.open_structure_dialog)

    def open_env_dialog(self):
        dlg = EnvironmentDialog(self.params, self)
        if dlg.exec():
            self.params = dlg.get_params()
            self.widget_3d.params = self.params

    def on_save_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "保存文件", "", "Log Files (*.csv *.txt);;All Files (*)")
        if file_path:
            display_name = file_path.split("/")[-1]
            self.btn_save_file.setText(f"文件: {display_name}")

    def start_simulation(self, t_total, dt):
        self.widget_ctrl.btn_run.setEnabled(False)
        self.widget_ctrl.btn_stop.setEnabled(True)
        self.widget_scope.reset_session()
        self.data_history = []

        log_settings = self.widget_ctrl.get_log_config()
        log_config = LogConfig(**log_settings)
        self.worker = SimulationWorker(self.params, t_total, dt, log_config=log_config, log_base_dir=".")
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
        self.widget_scope.update_data(data)
        self.live_panel.update_values(data)
        self.data_history.append(data)

        # 2. Update 3D View
        total_frames = len(self.data_history)
        self.widget_3d.update_timeline(total_frames, data['time'])

        if self.widget_3d.chk_sync.isChecked():
            self._update_3d_view(data)

        # 3. Update Power Label (Use updated key: env_GenPower)
        pwr = data.get('env_GenPower', data.get('GenPwr', 0.0))
        p_mw = pwr / 1e6
        self.widget_ctrl.lbl_power.setText(f"Power: {p_mw:.2f} MW")
        self.btn_power.setText(f"实施功率: {p_mw:.2f} MW")
        self.btn_runtime.setText(f"运行时间: {data['time']:.2f} s")

        self.observer_panel.update_data(data)
        channels = sorted(data.keys())
        self.observer_panel.set_available_channels(channels)

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
        history = self.data_history
        if 0 <= idx < len(history):
            data = history[idx]
            self._update_3d_view(data)
            self.widget_3d.lbl_time.setText(f"T: {data['time']:.2f}s")

    @Slot()
    def on_sim_finished(self):
        self.widget_ctrl.btn_run.setEnabled(True)
        self.widget_ctrl.btn_stop.setEnabled(False)
        self.widget_ctrl.progress.setValue(100)

        log_path = getattr(self.worker, 'log_dir', 'Unknown')
        QMessageBox.information(self, "Simulation Finished",
                                f"Simulation Completed.\nData saved to:\n{log_path}")

    def open_wind_dialog(self):
        dlg = WindDialog(self.params, self)
        if dlg.exec():
            dlg.apply()

    def open_wave_dialog(self):
        dlg = WaveCurrentDialog(self.params, self)
        if dlg.exec():
            dlg.apply()

    def open_mooring_dialog(self):
        dlg = MooringDialog(self.params, self)
        if dlg.exec():
            dlg.apply()

    def open_structure_dialog(self):
        dlg = StructureDialog(self.params, self)
        if dlg.exec():
            dlg.apply()
