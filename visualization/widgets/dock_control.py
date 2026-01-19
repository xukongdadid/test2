from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                               QProgressBar, QGroupBox, QTreeWidget, QTreeWidgetItem,
                               QDoubleSpinBox, QFormLayout, QComboBox, QHBoxLayout,
                               QDialog, QDialogButtonBox, QToolButton, QMenu)
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QTreeWidgetItemIterator
from visualization.save_config import SAVE_GROUPS, default_save_config

# --- [V3.1] 22-DOF Full Channel Config ---
CHANNEL_CONFIG = {
    # (A) Platform Rigid Body (6)
    "Platform Motion (刚体)": {
        "Surge (m)": "qSg", "Sway (m)": "qSw", "Heave (m)": "qHv",
        "Roll (deg)": "qR", "Pitch (deg)": "qP", "Yaw (deg)": "qY"
    },
    # (B) Tower Elastic (4)
    "Tower Elastic (塔筒柔性)": {
        "Fore-Aft Mode 1 (m)": "qTFA1", "Fore-Aft Mode 2 (m)": "qTFA2",
        "Side-Side Mode 1 (m)": "qTSS1", "Side-Side Mode 2 (m)": "qTSS2"
    },
    # (C) Nacelle & Drivetrain (3)
    "Nacelle & Drivetrain": {
        "Nacelle Yaw (deg)": "qyaw",
        "Gen Azimuth (deg)": "qGeAz",
        "Drivetrain Torsion (deg)": "qDrTr"
    },
    # (D) Blades (9)
    "Blade 1 Elastic": {
        "Flapwise Mode 1 (m)": "qB1F1", "Edgewise Mode 1 (m)": "qB1E1", "Flapwise Mode 2 (m)": "qB1F2"
    },
    "Blade 2 Elastic": {
        "Flapwise Mode 1 (m)": "qB2F1", "Edgewise Mode 1 (m)": "qB2E1", "Flapwise Mode 2 (m)": "qB2F2"
    },
    "Blade 3 Elastic": {
        "Flapwise Mode 1 (m)": "qB3F1", "Edgewise Mode 1 (m)": "qB3E1", "Flapwise Mode 2 (m)": "qB3F2"
    },
    # External Environments
    "Environment": {
        "Wind Speed (m/s)": "env_wind_speed",
        "Wave Elevation (m)": "env_wave_elev",
        "Thrust (N)": "thrust", "Gen Torque (Nm)": "gen_torque"
    }
}

class ControlDockWidget(QWidget):
    sig_start_sim = Signal(float, float) 
    sig_stop_sim = Signal()
    sig_open_env = Signal()
    sig_channels_changed = Signal(list, list) 
    sig_save_config_changed = Signal(dict)

    def __init__(self, parent=None, show_channels=False, show_env_button=False):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.show_channels = show_channels
        self.show_env_button = show_env_button
        
        # --- Simulation Control ---
        grp_sim = QGroupBox("Simulation Control")
        layout_sim = QVBoxLayout()
        
        form_time = QFormLayout()
        self.spin_time = QDoubleSpinBox(); self.spin_time.setRange(10, 3600); self.spin_time.setValue(600); self.spin_time.setSuffix(" s")
        self.spin_dt = QDoubleSpinBox(); self.spin_dt.setRange(0.001, 0.1); self.spin_dt.setValue(0.01); self.spin_dt.setSingleStep(0.005); self.spin_dt.setDecimals(3); self.spin_dt.setSuffix(" s")
        form_time.addRow("Total Time:", self.spin_time)
        form_time.addRow("Time Step:", self.spin_dt)
        layout_sim.addLayout(form_time)
        
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Start"); self.btn_run.clicked.connect(self.on_start_clicked)
        self.btn_stop = QPushButton("Stop"); self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self.sig_stop_sim.emit)
        self.btn_env = QPushButton("Env..."); self.btn_env.clicked.connect(self.sig_open_env.emit)
        btn_layout.addWidget(self.btn_run); btn_layout.addWidget(self.btn_stop)
        if self.show_env_button:
            btn_layout.addWidget(self.btn_env)
        layout_sim.addLayout(btn_layout)
        
        # Progress Bar
        self.progress = QProgressBar()
        layout_sim.addWidget(self.progress)
        
        # Timer Label
        self.lbl_timer = QLabel("Elapsed: 00:00")
        self.lbl_timer.setAlignment(Qt.AlignCenter)
        layout_sim.addWidget(self.lbl_timer)
        
        # [Fix] Power Label (Added back)
        self.lbl_power = QLabel("Power: 0.00 MW")
        self.lbl_power.setAlignment(Qt.AlignCenter)
        self.lbl_power.setStyleSheet("font-weight: bold; font-size: 14px; color: #4CAF50;")
        layout_sim.addWidget(self.lbl_power)

        # Live Telemetry
        grp_tel = QGroupBox("Live Telemetry")
        layout_tel = QVBoxLayout()
        self.lbl_dof = QLabel("DOF: Surge 0.00 | Sway 0.00 | Heave 0.00")
        self.lbl_env = QLabel("Env: Wind 0.0 m/s | Wave 0.0 m")
        self.lbl_moor = QLabel("Moor: Fx 0.0 | Fy 0.0 | Fz 0.0")
        layout_tel.addWidget(self.lbl_dof)
        layout_tel.addWidget(self.lbl_env)
        layout_tel.addWidget(self.lbl_moor)
        grp_tel.setLayout(layout_tel)
        self.layout.addWidget(grp_tel)
        
        grp_sim.setLayout(layout_sim)
        self.layout.addWidget(grp_sim)
        
        # --- 22-DOF Channels (Optional) ---
        self.tree = None
        if self.show_channels:
            grp_chan = QGroupBox("22-DOF Output Selection")
            layout_chan = QVBoxLayout()
            self.tree = QTreeWidget()
            self.tree.setHeaderLabels(["Channel", "Color", "Style"])
            self.tree.setColumnWidth(0, 200)
            self.tree.itemChanged.connect(self.on_item_changed)
            layout_chan.addWidget(self.tree)
            grp_chan.setLayout(layout_chan)
            self.layout.addWidget(grp_chan)

            self._init_tree()
            self.selected_keys = []

        # --- Data Save Selection ---
        grp_save = QGroupBox("Data Save")
        layout_save = QHBoxLayout()
        self.btn_save_cfg = QToolButton()
        self.btn_save_cfg.setText("Save Channels")
        self.btn_save_cfg.setPopupMode(QToolButton.InstantPopup)
        save_menu = QMenu(self.btn_save_cfg)
        act_cfg = save_menu.addAction("Configure...")
        act_reset = save_menu.addAction("Reset Default")
        act_cfg.triggered.connect(self.open_save_dialog)
        act_reset.triggered.connect(self.reset_save_config)
        self.btn_save_cfg.setMenu(save_menu)
        self.lbl_save_hint = QLabel("Default: all groups")
        self.lbl_save_hint.setStyleSheet("color: #7a7a7a;")
        layout_save.addWidget(self.btn_save_cfg)
        layout_save.addWidget(self.lbl_save_hint)
        layout_save.addStretch()
        grp_save.setLayout(layout_save)
        self.layout.addWidget(grp_save)

        self.save_config = default_save_config()
        self.layout.addStretch()

    def _init_tree(self):
        self.tree.clear()
        self._add_tree_items(self.tree, CHANNEL_CONFIG)
        self.tree.expandAll()

    def _add_tree_items(self, parent, config):
        for name, val in config.items():
            item = QTreeWidgetItem(parent)
            item.setText(0, name)
            
            if isinstance(val, dict):
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._add_tree_items(item, val)
            else:
                item.setData(0, 100, val) # Key
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Unchecked)
                
                # Style Combos
                combo_color = QComboBox()
                combo_color.addItems(['Red', 'Green', 'Blue', 'Cyan', 'Magenta', 'Yellow', 'White'])
                self.tree.setItemWidget(item, 1, combo_color)
                combo_color.currentIndexChanged.connect(lambda: self.on_item_changed(None, 0))

    def on_start_clicked(self):
        self.sig_start_sim.emit(self.spin_time.value(), self.spin_dt.value())

    def on_item_changed(self, item, col):
        if not self.tree:
            return
        channels = []
        it = QTreeWidgetItemIterator(self.tree)
        while it.value():
            item = it.value()
            if item.checkState(0) == Qt.Checked and item.data(0, 100):
                key = item.data(0, 100)
                w_color = self.tree.itemWidget(item, 1)
                color = w_color.currentText().lower() if w_color else 'red'
                channels.append({'key': key, 'name': item.text(0), 'color': color, 'style': 'Solid'})
            it += 1
        self.sig_channels_changed.emit(channels, []) 

    def open_save_dialog(self):
        dlg = DataSaveDialog(self.save_config, self)
        if dlg.exec():
            self.save_config = dlg.get_selection()
            picked = sum(len(v) for v in self.save_config.values())
            self.lbl_save_hint.setText(f"Selected fields: {picked}")
            self.sig_save_config_changed.emit(self.save_config)

    def reset_save_config(self):
        self.save_config = default_save_config()
        self.lbl_save_hint.setText("Default: all groups")
        self.sig_save_config_changed.emit(self.save_config)
    
    def update_progress(self, percent, elapsed_sec):
        self.progress.setValue(int(percent))
        mins, secs = divmod(int(elapsed_sec), 60)
        self.lbl_timer.setText(f"Elapsed: {mins:02d}:{secs:02d}")

    def update_telemetry(self, data):
        surge = data.get("dof_Surge", 0.0)
        sway = data.get("dof_Sway", 0.0)
        heave = data.get("dof_Heave", 0.0)
        wind = data.get("env_wind_speed", data.get("WindSpeed", 0.0))
        wave = data.get("env_wave_elev", data.get("WaveElev", 0.0))
        moor_fx = data.get("MoorFx", 0.0)
        moor_fy = data.get("MoorFy", 0.0)
        moor_fz = data.get("MoorFz", 0.0)
        self.lbl_dof.setText(f"DOF: Surge {surge:.2f} | Sway {sway:.2f} | Heave {heave:.2f}")
        self.lbl_env.setText(f"Env: Wind {wind:.2f} m/s | Wave {wave:.2f} m")
        self.lbl_moor.setText(f"Moor: Fx {moor_fx:.2f} | Fy {moor_fy:.2f} | Fz {moor_fz:.2f}")


class DataSaveDialog(QDialog):
    def __init__(self, current_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Save Configuration")
        self.resize(420, 420)
        self.layout = QVBoxLayout(self)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Group / Field"])
        self.layout.addWidget(self.tree)

        self._populate_tree(current_config)
        self.tree.expandAll()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        self.layout.addWidget(btns)

    def _populate_tree(self, current_config):
        self.tree.clear()
        for group, fields in SAVE_GROUPS.items():
            g_item = QTreeWidgetItem(self.tree)
            g_item.setText(0, group)
            g_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            selected_fields = set(current_config.get(group, []))
            g_item.setCheckState(0, Qt.Checked if selected_fields else Qt.Unchecked)
            for field in fields:
                f_item = QTreeWidgetItem(g_item)
                f_item.setText(0, field)
                f_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                f_item.setCheckState(0, Qt.Checked if field in selected_fields else Qt.Unchecked)

    def get_selection(self):
        selection = {group: [] for group in SAVE_GROUPS.keys()}
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            g_item = root.child(i)
            group = g_item.text(0)
            for j in range(g_item.childCount()):
                f_item = g_item.child(j)
                if f_item.checkState(0) == Qt.Checked:
                    selection[group].append(f_item.text(0))
        return selection
