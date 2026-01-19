from PySide6.QtWidgets import (QMainWindow, QMessageBox, QWidget, QVBoxLayout,
                               QHBoxLayout, QComboBox, QSplitter)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction

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

        mode_bar = QHBoxLayout()
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Operator Mode", "Observer Mode"])
        self.mode_selector.currentIndexChanged.connect(self.on_mode_changed)
        mode_bar.addWidget(self.mode_selector)
        mode_bar.addStretch()
        main_layout.addLayout(mode_bar)

        self.stack = QWidget()
        stack_layout = QVBoxLayout(self.stack)
        self.stack_layout = stack_layout

        self.operator_widget = self._build_operator_view()
        self.observer_widget = self._build_observer_view()

        stack_layout.addWidget(self.operator_widget)
        stack_layout.addWidget(self.observer_widget)
        self.observer_widget.hide()
        main_layout.addWidget(self.stack)

        self.setCentralWidget(central)

    def _build_operator_view(self):
        container = QWidget()
        layout = QHBoxLayout(container)
        splitter = QSplitter(Qt.Horizontal)

        self.widget_3d = View3DDockWidget(self.params)
        self.widget_3d.sig_frame_request.connect(self.on_3d_frame_request)
        self.widget_3d.sig_edit_wind.connect(self.open_wind_dialog)
        self.widget_3d.sig_edit_wave.connect(self.open_wave_dialog)
        self.widget_3d.sig_edit_mooring.connect(self.open_mooring_dialog)
        self.widget_3d.sig_edit_structure.connect(self.open_structure_dialog)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.widget_ctrl = ControlDockWidget()
        self.widget_scope = ScopeDockWidget()
        self.live_panel = LiveDataPanel()

        right_layout.addWidget(self.widget_ctrl)
        right_layout.addWidget(self.live_panel)
        right_layout.addWidget(self.widget_scope)

        splitter.addWidget(self.widget_3d)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        self.widget_ctrl.sig_start_sim.connect(self.start_simulation)
        self.widget_ctrl.sig_stop_sim.connect(self.stop_simulation)
        self.widget_ctrl.sig_open_env.connect(self.open_env_dialog)
        self.widget_ctrl.sig_channels_changed.connect(self.widget_scope.update_channels)

        return container

    def _build_observer_view(self):
        container = QWidget()
        layout = QHBoxLayout(container)
        splitter = QSplitter(Qt.Horizontal)

        self.observer_3d = View3DDockWidget(self.params)
        self.observer_panel = ObserverPanel()

        splitter.addWidget(self.observer_3d)
        splitter.addWidget(self.observer_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        return container

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
            self.observer_3d.params = self.params

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
        self.observer_3d.update_timeline(total_frames, data['time'])

        if self.widget_3d.chk_sync.isChecked():
            self._update_3d_view(data)
        if self.observer_3d.chk_sync.isChecked():
            self._update_3d_view(data, target=self.observer_3d)

        # 3. Update Power Label (Use updated key: env_GenPower)
        pwr = data.get('env_GenPower', data.get('GenPwr', 0.0))
        p_mw = pwr / 1e6
        self.widget_ctrl.lbl_power.setText(f"Power: {p_mw:.2f} MW")

        self.observer_panel.update_data(data)
        if self.observer_widget.isVisible():
            channels = sorted(data.keys())
            self.observer_panel.set_available_channels(channels)

    def _update_3d_view(self, data, target=None):
        """Map new dof_ keys to 3D view"""
        state_rigid = [
            data.get('dof_Surge', 0),
            data.get('dof_Sway', 0),
            data.get('dof_Heave', 0),
            data.get('dof_Roll', 0),
            data.get('dof_Pitch', 0),
            data.get('dof_Yaw', 0)
        ]
        view = target or self.widget_3d
        view.update_pose(state_rigid, data['time'])

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

    def on_mode_changed(self, idx):
        if idx == 0:
            self.operator_widget.show()
            self.observer_widget.hide()
        else:
            self.operator_widget.hide()
            self.observer_widget.show()
            channels = sorted({k for entry in self.data_history for k in entry.keys()})
            self.observer_panel.set_available_channels(channels)

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
