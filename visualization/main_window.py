from PySide6.QtWidgets import (QMainWindow, QMessageBox, QWidget, QVBoxLayout,
                               QHBoxLayout, QFrame, QPushButton, QFileDialog,
                               QLabel, QDoubleSpinBox, QSplitter)
from PySide6.QtCore import Slot, QEvent

from config.fast_params import SystemParams
from solver.worker import SimulationWorker
from visualization.widgets.dock_3d import View3DDockWidget
from visualization.dialog_env import EnvironmentDialog
from visualization.edit_dialogs import WindDialog, WaveCurrentDialog, MooringDialog, StructureDialog
from visualization.widgets.observer_panel import ObserverPanel
from solver.data_logger import LogConfig


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"LabFAST V0.0.1 (AeroDyn 15 + Kane 22-DOF)")
        self.resize(1600, 1000)

        self.params = SystemParams()
        self.worker = None
        self.log_base_dir = "."
        self.mode = "operator"

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
        self.widget_3d.view.installEventFilter(self)

        if hasattr(self, "_build_operator_view"):
            self.operator_widget = self._build_operator_view()
        else:
            self.operator_widget = self._build_operator_bar()
        self.observer_widget = self._build_observer_view()
        self.observer_widget.hide()

        main_layout.addWidget(self.operator_widget, stretch=1)
        main_layout.addWidget(self.observer_widget, stretch=1)

        self.setCentralWidget(central)

    def _build_operator_view(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self.widget_3d, stretch=8)

        bottom_bar = self._build_operator_bar()
        layout.addWidget(bottom_bar, stretch=2)
        return container

    def _build_operator_bar(self):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setSpacing(16)
        layout.setContentsMargins(0, 8, 0, 0)

        data_panel = self._build_data_panel()
        control_panel = self._build_micro_control_panel()
        power_panel = self._build_power_panel()

        layout.addWidget(data_panel, stretch=2)
        layout.addWidget(control_panel, stretch=4)
        layout.addWidget(power_panel, stretch=2)
        return container

    def _build_data_panel(self):
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
        self.btn_save_file = QPushButton("数据保存通道")
        self.btn_save_file.clicked.connect(self.on_save_file)
        layout.addWidget(self.btn_save_file)
        layout.addStretch()
        return frame

    def _build_micro_control_panel(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            "QFrame {"
            "background-color: #0b0f1a;"
            "border: 1px solid #19d3ff;"
            "border-radius: 10px;"
            "}"
            "QLabel {"
            "color: #d7faff;"
            "}"
            "QDoubleSpinBox {"
            "background-color: #0f1c2c;"
            "color: #d7faff;"
            "border: 1px solid #19d3ff;"
            "border-radius: 4px;"
            "padding: 2px 6px;"
            "}"
            "QPushButton {"
            "color: #d7faff;"
            "background-color: #0f1c2c;"
            "border: 1px solid #19d3ff;"
            "border-radius: 6px;"
            "padding: 8px 12px;"
            "font-weight: 600;"
            "}"
            "QPushButton:hover {"
            "background-color: #12304a;"
            "}"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("微控制板")
        header.setStyleSheet("font-weight: 700; font-size: 14px;")
        layout.addWidget(header)

        row_time = QHBoxLayout()
        row_time.addWidget(QLabel("时长(s):"))
        self.spin_time = QDoubleSpinBox()
        self.spin_time.setRange(10, 3600)
        self.spin_time.setValue(600)
        row_time.addWidget(self.spin_time)
        row_time.addStretch()
        layout.addLayout(row_time)

        row_step = QHBoxLayout()
        row_step.addWidget(QLabel("步长(s):"))
        self.spin_dt = QDoubleSpinBox()
        self.spin_dt.setRange(0.001, 0.1)
        self.spin_dt.setDecimals(3)
        self.spin_dt.setSingleStep(0.005)
        self.spin_dt.setValue(0.01)
        row_step.addWidget(self.spin_dt)
        row_step.addStretch()
        layout.addLayout(row_step)

        row_btn = QHBoxLayout()
        self.btn_run = QPushButton("开始")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_run.clicked.connect(self.on_micro_start)
        self.btn_stop.clicked.connect(self.stop_simulation)
        row_btn.addWidget(self.btn_run)
        row_btn.addWidget(self.btn_stop)
        layout.addLayout(row_btn)
        return frame

    def _build_power_panel(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            "QFrame {"
            "background-color: #070b14;"
            "border: 1px solid #7cf4ff;"
            "border-radius: 10px;"
            "}"
            "QLabel {"
            "color: #d7faff;"
            "}"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        self.lbl_power = QLabel("功率: 0.00 MW")
        self.lbl_power.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.lbl_runtime = QLabel("运行时间: 0.00 s")
        layout.addStretch()
        layout.addWidget(self.lbl_power)
        layout.addWidget(self.lbl_runtime)
        layout.addStretch()
        return frame

    def _build_observer_view(self):
        container = QWidget()
        layout = QHBoxLayout(container)
        splitter = QSplitter()
        self.observer_3d = View3DDockWidget(self.params)
        self.observer_3d.view.installEventFilter(self)
        self.observer_3d.sig_frame_request.connect(self.on_observer_frame_request)
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

    def on_save_file(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据保存通道", self.log_base_dir)
        if folder:
            self.log_base_dir = folder
            display_name = folder.rstrip("/").split("/")[-1] or folder
            self.btn_save_file.setText(f"数据保存: {display_name}")

    def on_micro_start(self):
        self.start_simulation(self.spin_time.value(), self.spin_dt.value())

    def start_simulation(self, t_total, dt):
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.data_history = []

        log_config = LogConfig(save_dof=True, save_hydro=True, save_wind_aero=True, save_moor=True)
        self.worker = SimulationWorker(self.params, t_total, dt, log_config=log_config, log_base_dir=self.log_base_dir)
        self.worker.data_signal.connect(self.on_data_update)
        self.worker.finished_signal.connect(self.on_sim_finished)
        self.worker.error_signal.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.start()

    def stop_simulation(self):
        if self.worker:
            self.worker.stop()
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    @Slot(dict)
    def on_data_update(self, data):
        # 1. Update 3D View
        self.data_history.append(data)
        total_frames = len(self.data_history)
        self.widget_3d.update_timeline(total_frames, data['time'])

        if self.widget_3d.chk_sync.isChecked():
            self._update_3d_view(self.widget_3d, data)
        if self.mode == "observer":
            self.observer_3d.update_timeline(total_frames, data['time'])
            if self.observer_3d.chk_sync.isChecked():
                self._update_3d_view(self.observer_3d, data)

        # 2. Update Power Label (Use updated key: env_GenPower)
        pwr = data.get('env_GenPower', data.get('GenPwr', 0.0))
        p_mw = pwr / 1e6
        self.lbl_power.setText(f"功率: {p_mw:.2f} MW")
        self.lbl_runtime.setText(f"运行时间: {data['time']:.2f} s")

        self.observer_panel.update_data(data)
        channels = sorted(data.keys())
        self.observer_panel.set_available_channels(channels)

    def _update_3d_view(self, view, data):
        """Map new dof_ keys to 3D view"""
        state_rigid = [
            data.get('dof_Surge', 0),
            data.get('dof_Sway', 0),
            data.get('dof_Heave', 0),
            data.get('dof_Roll', 0),
            data.get('dof_Pitch', 0),
            data.get('dof_Yaw', 0)
        ]
        view.update_pose(state_rigid, data['time'])

    @Slot(int)
    def on_3d_frame_request(self, idx):
        history = self.data_history
        if 0 <= idx < len(history):
            data = history[idx]
            self._update_3d_view(self.widget_3d, data)
            self.widget_3d.lbl_time.setText(f"T: {data['time']:.2f}s")

    @Slot(int)
    def on_observer_frame_request(self, idx):
        history = self.data_history
        if 0 <= idx < len(history):
            data = history[idx]
            self._update_3d_view(self.observer_3d, data)
            self.observer_3d.lbl_time.setText(f"T: {data['time']:.2f}s")

    @Slot()
    def on_sim_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

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

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonDblClick:
            self.toggle_mode()
            return True
        return super().eventFilter(obj, event)

    def toggle_mode(self):
        if self.mode == "operator":
            self.mode = "observer"
            self.operator_widget.hide()
            self.observer_widget.show()
            if self.data_history:
                self._update_3d_view(self.observer_3d, self.data_history[-1])
        else:
            self.mode = "operator"
            self.observer_widget.hide()
            self.operator_widget.show()
