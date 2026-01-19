from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QStackedWidget, QMessageBox)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont

from config.fast_params import SystemParams
from solver.worker import SimulationWorker
from visualization.widgets.dock_control import ControlDockWidget
from visualization.widgets.dock_3d import View3DDockWidget
from visualization.widgets.panel_observer import ObserverPanel
from visualization.widgets.scope_window import TimeScopeWindow, FrequencyScopeWindow
from visualization.dialog_env import EnvironmentDialog
from visualization.dialog_turbine import TurbineDialog
from visualization.data_saver import DataSaver


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LabFAST V0.0.1 (AeroDyn 15 + Kane 22-DOF)")
        self.resize(1600, 1000)

        self.params = SystemParams()
        self.worker = None
        self.data_history = []
        self.time_scopes = []
        self.freq_scopes = []
        self.data_saver = DataSaver()

        self._build_ui()
        self.apply_theme()
        self.set_mode("operator")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 12, 16, 12)

        top_bar = QHBoxLayout()
        title = QLabel("LabFAST Digital Twin")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        top_bar.addWidget(title)

        top_bar.addStretch()
        self.btn_operator = QPushButton("Operator")
        self.btn_observer = QPushButton("Observer")
        for btn in (self.btn_operator, self.btn_observer):
            btn.setCheckable(True)
            btn.setMinimumWidth(100)
        self.btn_operator.clicked.connect(lambda: self.set_mode("operator"))
        self.btn_observer.clicked.connect(lambda: self.set_mode("observer"))
        top_bar.addWidget(self.btn_operator)
        top_bar.addWidget(self.btn_observer)
        layout.addLayout(top_bar)

        body = QHBoxLayout()
        layout.addLayout(body)

        self.widget_3d = View3DDockWidget(self.params)
        body.addWidget(self.widget_3d, stretch=3)

        self.side_stack = QStackedWidget()
        self.side_stack.setFixedWidth(360)
        body.addWidget(self.side_stack, stretch=0)

        self.operator_panel = ControlDockWidget(show_channels=False, show_env_button=False)
        self.observer_panel = ObserverPanel()
        self.side_stack.addWidget(self.operator_panel)
        self.side_stack.addWidget(self.observer_panel)

        # --- Signal Connections ---
        self.operator_panel.sig_start_sim.connect(self.start_simulation)
        self.operator_panel.sig_stop_sim.connect(self.stop_simulation)
        self.operator_panel.sig_save_config_changed.connect(self.on_save_config_changed)

        self.observer_panel.sig_create_time_scopes.connect(self.create_time_scopes)
        self.observer_panel.sig_create_freq_scopes.connect(self.create_freq_scopes)

        self.widget_3d.sig_frame_request.connect(self.on_3d_frame_request)
        self.widget_3d.sig_edit_wind.connect(lambda: self.open_env_dialog("wind"))
        self.widget_3d.sig_edit_wave.connect(lambda: self.open_env_dialog("wave"))
        self.widget_3d.sig_edit_mooring.connect(lambda: self.open_env_dialog("mooring"))
        self.widget_3d.sig_edit_structure.connect(self.open_structure_dialog)

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #f4f7fb; }
            QLabel { color: #1f2933; }
            QGroupBox {
                border: 1px solid #d7dde3;
                border-radius: 10px;
                margin-top: 12px;
                font-weight: 600;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #374151;
            }
            QPushButton {
                background: #1f6feb;
                color: white;
                border-radius: 8px;
                padding: 6px 12px;
            }
            QPushButton:checked {
                background: #0f4bb3;
            }
            QPushButton:disabled {
                background: #b9c0c6;
                color: #eef1f4;
            }
            QProgressBar {
                border: 1px solid #d7dde3;
                border-radius: 6px;
                text-align: center;
            }
            QProgressBar::chunk { background: #46b6a1; }
        """)

    def set_mode(self, mode):
        is_operator = mode == "operator"
        self.btn_operator.setChecked(is_operator)
        self.btn_observer.setChecked(not is_operator)
        self.side_stack.setCurrentIndex(0 if is_operator else 1)

    def open_env_dialog(self, tab=None):
        dlg = EnvironmentDialog(self.params, self, initial_tab=tab)
        if dlg.exec():
            self.params = dlg.get_params()
            self.widget_3d.apply_params(self.params)

    def open_structure_dialog(self):
        dlg = TurbineDialog(self.params, self)
        if dlg.exec():
            self.params = dlg.get_params()
            self.widget_3d.apply_params(self.params)

    def create_time_scopes(self, count):
        for _ in range(count):
            win = TimeScopeWindow(self)
            win.configure_channels()
            if not win.channels:
                win.close()
                continue
            self._register_scope(win, self.time_scopes)
            win.show()

    def create_freq_scopes(self, count):
        for _ in range(count):
            win = FrequencyScopeWindow(self)
            win.configure_channels()
            if not win.channels:
                win.close()
                continue
            self._register_scope(win, self.freq_scopes)
            win.show()

    def _register_scope(self, win, collection):
        collection.append(win)
        win.finished.connect(lambda _: self._unregister_scope(win, collection))

    def _unregister_scope(self, win, collection):
        if win in collection:
            collection.remove(win)

    def start_simulation(self, t_total, dt):
        self.operator_panel.btn_run.setEnabled(False)
        self.operator_panel.btn_stop.setEnabled(True)
        self.data_history = []
        for win in self.time_scopes + self.freq_scopes:
            win.reset_session()

        self.data_saver.start(self.operator_panel.save_config)

        self.worker = SimulationWorker(self.params, t_total, dt)
        self.worker.data_signal.connect(self.on_data_update)
        self.worker.progress_signal.connect(self.operator_panel.update_progress)
        self.worker.finished_signal.connect(self.on_sim_finished)
        self.worker.error_signal.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.start()

    def stop_simulation(self):
        if self.worker:
            self.worker.stop()

    def on_save_config_changed(self, config):
        if self.data_saver.folder:
            self.data_saver.start(config)

    @Slot(dict)
    def on_data_update(self, data):
        self.data_history.append(data)
        self.data_saver.write(data)

        for win in self.time_scopes:
            win.update_data(data)
        for win in self.freq_scopes:
            win.update_data(data)

        total_frames = len(self.data_history)
        self.widget_3d.update_timeline(total_frames, data["time"])
        self.widget_3d.update_environment(data)

        if self.widget_3d.chk_sync.isChecked():
            self._update_3d_view(data)

        pwr = data.get("GenPwr", data.get("env_GenPower", 0.0))
        self.operator_panel.lbl_power.setText(f"Power: {pwr / 1e6:.2f} MW")
        self.operator_panel.update_telemetry(data)

    def _update_3d_view(self, data):
        state_rigid = [
            data.get("dof_Surge", 0),
            data.get("dof_Sway", 0),
            data.get("dof_Heave", 0),
            data.get("dof_Roll", 0),
            data.get("dof_Pitch", 0),
            data.get("dof_Yaw", 0),
        ]
        self.widget_3d.update_pose(state_rigid, data["time"])

    @Slot(int)
    def on_3d_frame_request(self, idx):
        if 0 <= idx < len(self.data_history):
            data = self.data_history[idx]
            self._update_3d_view(data)
            self.widget_3d.lbl_time.setText(f"T: {data['time']:.2f}s")

    @Slot()
    def on_sim_finished(self):
        self.operator_panel.btn_run.setEnabled(True)
        self.operator_panel.btn_stop.setEnabled(False)
        self.operator_panel.progress.setValue(100)
        log_path = self.data_saver.folder or "Unknown"
        self.data_saver.close()
        QMessageBox.information(self, "Simulation Finished",
                                f"Simulation Completed.\nData saved to:\n{log_path}")
