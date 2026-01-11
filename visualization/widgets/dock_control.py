from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                               QProgressBar, QGroupBox, QTreeWidget, QTreeWidgetItem,
                               QDoubleSpinBox, QFormLayout, QComboBox, QHBoxLayout)
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QTreeWidgetItemIterator

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
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
        btn_layout.addWidget(self.btn_run); btn_layout.addWidget(self.btn_stop); btn_layout.addWidget(self.btn_env)
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
        
        grp_sim.setLayout(layout_sim)
        self.layout.addWidget(grp_sim)
        
        # --- 22-DOF Channels ---
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
    
    def update_progress(self, percent, elapsed_sec):
        self.progress.setValue(int(percent))
        mins, secs = divmod(int(elapsed_sec), 60)
        self.lbl_timer.setText(f"Elapsed: {mins:02d}:{secs:02d}")