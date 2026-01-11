from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QPushButton, QDoubleSpinBox, QLabel, QGroupBox,
                               QComboBox, QTabWidget, QWidget, QFileDialog, QLineEdit)
from PySide6.QtCore import Qt


class EnvironmentDialog(QDialog):
    def __init__(self, current_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LabFAST V0.0.1 Environment Config")
        self.resize(600, 500)
        self.params = current_params

        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Initialize Tabs
        self.init_wind_tab()
        self.init_wave_tab()
        self.init_current_tab()
        self.init_hydro_tab()
        self.init_mooring_tab()  # [New] 添加系泊标签页

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("Apply");
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("Cancel");
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_ok);
        btn_layout.addWidget(self.btn_cancel)
        self.layout.addLayout(btn_layout)

        self.load_params()

    def init_wind_tab(self):
        self.tab_wind = QWidget()
        layout = QVBoxLayout(self.tab_wind)

        grp_type = QGroupBox("Wind Source")
        form_type = QFormLayout()
        self.combo_wind_type = QComboBox()
        self.combo_wind_type.addItems(
            ["Steady (Constant)", "Internal Turbulent (Kaimal approx)", "External (TurSim .bts)"])
        self.combo_wind_type.currentIndexChanged.connect(self.on_wind_type_changed)
        form_type.addRow("Model Type:", self.combo_wind_type)
        grp_type.setLayout(form_type)
        layout.addWidget(grp_type)

        grp_param = QGroupBox("Parameters")
        form_param = QFormLayout()
        self.spin_wind_speed = QDoubleSpinBox();
        self.spin_wind_speed.setRange(0, 50);
        self.spin_wind_speed.setSuffix(" m/s")
        form_param.addRow("Ref. Speed (Hub):", self.spin_wind_speed)
        self.spin_shear = QDoubleSpinBox();
        self.spin_shear.setRange(0, 1);
        self.spin_shear.setSingleStep(0.01)
        form_param.addRow("Shear Exp:", self.spin_shear)
        self.line_wind_file = QLineEdit()
        self.btn_browse = QPushButton("Browse...");
        self.btn_browse.clicked.connect(self.browse_file)
        h_file = QHBoxLayout();
        h_file.addWidget(self.line_wind_file);
        h_file.addWidget(self.btn_browse)
        form_param.addRow("BTS File:", h_file)
        grp_param.setLayout(form_param)
        layout.addWidget(grp_param)

        layout.addStretch()
        self.tabs.addTab(self.tab_wind, "Wind")

    def init_wave_tab(self):
        self.tab_wave = QWidget()
        layout = QVBoxLayout(self.tab_wave)

        grp_wave = QGroupBox("Sea State (Waves)")
        form = QFormLayout()

        self.combo_wave_mod = QComboBox()
        self.combo_wave_mod.addItems(
            ["0: Still Water", "1: Regular (Airy)", "2: Irregular (JONSWAP)", "3: Irregular (PM)"])
        form.addRow("Wave Model:", self.combo_wave_mod)

        self.spin_wave_h = QDoubleSpinBox();
        self.spin_wave_h.setRange(0, 20);
        self.spin_wave_h.setSuffix(" m")
        form.addRow("Hs (Sig. Height):", self.spin_wave_h)
        self.spin_wave_t = QDoubleSpinBox();
        self.spin_wave_t.setRange(0, 30);
        self.spin_wave_t.setSuffix(" s")
        form.addRow("Tp (Peak Period):", self.spin_wave_t)
        self.spin_wave_gamma = QDoubleSpinBox();
        self.spin_wave_gamma.setRange(1, 10);
        self.spin_wave_gamma.setValue(3.3)
        form.addRow("Gamma (Peak Shape):", self.spin_wave_gamma)

        grp_wave.setLayout(form)
        layout.addWidget(grp_wave)
        layout.addStretch()
        self.tabs.addTab(self.tab_wave, "Wave")

    def init_current_tab(self):
        self.tab_curr = QWidget()
        layout = QVBoxLayout(self.tab_curr)

        grp_c = QGroupBox("Sea Current Profile")
        form = QFormLayout()

        self.spin_curr_ref = QDoubleSpinBox();
        self.spin_curr_ref.setRange(-5, 5);
        self.spin_curr_ref.setSuffix(" m/s")
        form.addRow("Uniform Velocity:", self.spin_curr_ref)

        self.spin_curr_near = QDoubleSpinBox();
        self.spin_curr_near.setRange(0, 5);
        self.spin_curr_near.setSuffix(" m/s")
        form.addRow("Near-Surface Vel (z=0):", self.spin_curr_near)

        self.spin_curr_sub = QDoubleSpinBox();
        self.spin_curr_sub.setRange(0, 5);
        self.spin_curr_sub.setSuffix(" m/s")
        form.addRow("Sub-Surface Vel (z=0):", self.spin_curr_sub)

        self.spin_curr_exp = QDoubleSpinBox();
        self.spin_curr_exp.setRange(0, 1);
        self.spin_curr_exp.setSingleStep(0.01)
        form.addRow("Power Law Exp (1/7):", self.spin_curr_exp)

        self.spin_ref_depth = QDoubleSpinBox();
        self.spin_ref_depth.setRange(0, 200);
        self.spin_ref_depth.setSuffix(" m")
        form.addRow("Ref. Depth:", self.spin_ref_depth)

        grp_c.setLayout(form)
        layout.addWidget(grp_c)
        layout.addStretch()
        self.tabs.addTab(self.tab_curr, "Current")

    def init_hydro_tab(self):
        self.tab_hydro = QWidget()
        layout = QVBoxLayout(self.tab_hydro)

        grp = QGroupBox("Hydrodynamics Theory")
        form = QFormLayout()

        self.combo_hydro = QComboBox()
        self.combo_hydro.addItems(["Strip Theory (Morison)", "Potential Flow (Linear)", "Hybrid (Pot + Viscous)"])
        form.addRow("Method:", self.combo_hydro)

        self.spin_depth = QDoubleSpinBox();
        self.spin_depth.setRange(10, 1000);
        self.spin_depth.setSuffix(" m")
        form.addRow("Water Depth:", self.spin_depth)

        grp.setLayout(form)
        layout.addWidget(grp)
        layout.addStretch()
        self.tabs.addTab(self.tab_hydro, "Hydro")

    def init_mooring_tab(self):
        self.tab_moor = QWidget()
        layout = QVBoxLayout(self.tab_moor)

        grp = QGroupBox("Mooring System Solver")
        form = QFormLayout()

        self.combo_moor_model = QComboBox()
        # [Modified] 增加 MoorEmm 选项 (index 3)
        self.combo_moor_model.addItems([
            "0: Simple Solver (Original)",
            "1: MAP++ Catenary (Bottom Contact)",
            "2: MoorDyn (External C++)",
            "3: MoorEmm (Python Lumped Mass)"  # <--- 新增
        ])

        self.combo_moor_model.currentIndexChanged.connect(self.on_moor_model_changed)
        form.addRow("Solver Model:", self.combo_moor_model)

        self.spin_moor_cb = QDoubleSpinBox()
        self.spin_moor_cb.setRange(0.0, 1.0)
        self.spin_moor_cb.setSingleStep(0.1)
        form.addRow("Seabed Friction (Cb):", self.spin_moor_cb)

        grp.setLayout(form)
        layout.addWidget(grp)

        # [Modified] 更新提示信息
        lbl_tip = QLabel(
            "Note:\n- Simple: Quasi-static analytic.\n- MoorDyn: OpenFAST C++ Module.\n- MoorEmm: LabFAST Python Lumped Mass.")
        lbl_tip.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(lbl_tip)

        layout.addStretch()
        self.tabs.addTab(self.tab_moor, "Mooring")

    def on_wind_type_changed(self, idx):
        is_ext = (idx == 2)
        self.line_wind_file.setEnabled(is_ext)
        self.btn_browse.setEnabled(is_ext)
        self.spin_wind_speed.setEnabled(True)

    def on_moor_model_changed(self, idx):
        # 0 = Simple (Original), 1 = MAP++
        # Cb 仅用于 MAP++ 准静态接触模型
        self.spin_moor_cb.setEnabled(idx == 1)

    def browse_file(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open BTS", "", "BTS Files (*.bts)")
        if fn: self.line_wind_file.setText(fn)

    def load_params(self):
        # Wind
        self.combo_wind_type.setCurrentIndex(self.params.Env_WindType)
        self.spin_wind_speed.setValue(self.params.Env_WindSpeed)
        self.spin_shear.setValue(self.params.Env_ShearExp)
        self.line_wind_file.setText(self.params.Env_WindFile)
        # Wave
        self.combo_wave_mod.setCurrentIndex(self.params.Env_WaveMod)
        self.spin_wave_h.setValue(self.params.Env_WaveHeight)
        self.spin_wave_t.setValue(self.params.Env_WavePeriod)
        self.spin_wave_gamma.setValue(self.params.Env_WaveGamma)
        # Current
        self.spin_curr_ref.setValue(self.params.Curr_RefSpeed)
        self.spin_curr_near.setValue(self.params.Curr_NearSurfSpeed)
        self.spin_curr_sub.setValue(self.params.Curr_SubSpeed)
        self.spin_curr_exp.setValue(self.params.Curr_SubExp)
        self.spin_ref_depth.setValue(self.params.Curr_RefDepth)
        # Hydro
        self.combo_hydro.setCurrentIndex(self.params.Hydro_Method)
        self.spin_depth.setValue(self.params.WaterDepth)
        # Mooring [Updated]
        # 使用 getattr 防止旧的 params 对象没有该属性报错
        moor_type = getattr(self.params, "Moor_ModelType", 0)
        moor_cb = getattr(self.params, "Moor_CB", 0.5)
        self.combo_moor_model.setCurrentIndex(moor_type)
        self.spin_moor_cb.setValue(moor_cb)
        self.on_moor_model_changed(moor_type)  # Trigger UI state

        self.on_wind_type_changed(self.params.Env_WindType)

    def get_params(self):
        self.params.Env_WindType = self.combo_wind_type.currentIndex()
        self.params.Env_WindSpeed = self.spin_wind_speed.value()
        self.params.Env_ShearExp = self.spin_shear.value()
        self.params.Env_WindFile = self.line_wind_file.text()

        self.params.Env_WaveMod = self.combo_wave_mod.currentIndex()
        self.params.Env_WaveHeight = self.spin_wave_h.value()
        self.params.Env_WavePeriod = self.spin_wave_t.value()
        self.params.Env_WaveGamma = self.spin_wave_gamma.value()

        self.params.Curr_RefSpeed = self.spin_curr_ref.value()
        self.params.Curr_NearSurfSpeed = self.spin_curr_near.value()
        self.params.Curr_SubSpeed = self.spin_curr_sub.value()
        self.params.Curr_SubExp = self.spin_curr_exp.value()
        self.params.Curr_RefDepth = self.spin_ref_depth.value()

        self.params.Hydro_Method = self.combo_hydro.currentIndex()
        self.params.WaterDepth = self.spin_depth.value()

        # Mooring [Updated]
        self.params.Moor_ModelType = self.combo_moor_model.currentIndex()
        self.params.Moor_CB = self.spin_moor_cb.value()

        return self.params