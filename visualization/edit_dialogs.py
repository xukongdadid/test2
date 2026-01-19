from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox,
    QDoubleSpinBox, QComboBox, QLabel
)


class WindDialog(QDialog):
    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self.setWindowTitle("Wind Settings")
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.combo_type = QComboBox()
        self.combo_type.addItems(["Steady", "Internal Turbulent", "External BTS"])
        self.combo_type.setCurrentIndex(self.params.Env_WindType)
        form.addRow("Wind Model:", self.combo_type)

        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0, 50)
        self.spin_speed.setValue(self.params.Env_WindSpeed)
        self.spin_speed.setSuffix(" m/s")
        form.addRow("Wind Speed:", self.spin_speed)

        self.spin_shear = QDoubleSpinBox()
        self.spin_shear.setRange(0, 1)
        self.spin_shear.setSingleStep(0.01)
        self.spin_shear.setValue(self.params.Env_ShearExp)
        form.addRow("Shear Exp:", self.spin_shear)

        layout.addLayout(form)
        layout.addWidget(QLabel("Tip: Right-click the 3D view for quick access."))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply(self):
        self.params.Env_WindType = self.combo_type.currentIndex()
        self.params.Env_WindSpeed = self.spin_speed.value()
        self.params.Env_ShearExp = self.spin_shear.value()


class WaveCurrentDialog(QDialog):
    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self.setWindowTitle("Wave & Current Settings")
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.combo_wave_mod = QComboBox()
        self.combo_wave_mod.addItems(["Still Water", "Regular (Airy)", "Irregular (JONSWAP)", "Irregular (PM)"])
        self.combo_wave_mod.setCurrentIndex(self.params.Env_WaveMod)
        form.addRow("Wave Model:", self.combo_wave_mod)

        self.spin_wave_h = QDoubleSpinBox()
        self.spin_wave_h.setRange(0, 20)
        self.spin_wave_h.setValue(self.params.Env_WaveHeight)
        self.spin_wave_h.setSuffix(" m")
        form.addRow("Wave Height:", self.spin_wave_h)

        self.spin_wave_t = QDoubleSpinBox()
        self.spin_wave_t.setRange(0, 30)
        self.spin_wave_t.setValue(self.params.Env_WavePeriod)
        self.spin_wave_t.setSuffix(" s")
        form.addRow("Wave Period:", self.spin_wave_t)

        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(1, 10)
        self.spin_gamma.setValue(self.params.Env_WaveGamma)
        form.addRow("Gamma:", self.spin_gamma)

        self.spin_curr = QDoubleSpinBox()
        self.spin_curr.setRange(-5, 5)
        self.spin_curr.setValue(self.params.Curr_RefSpeed)
        self.spin_curr.setSuffix(" m/s")
        form.addRow("Current Speed:", self.spin_curr)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply(self):
        self.params.Env_WaveMod = self.combo_wave_mod.currentIndex()
        self.params.Env_WaveHeight = self.spin_wave_h.value()
        self.params.Env_WavePeriod = self.spin_wave_t.value()
        self.params.Env_WaveGamma = self.spin_gamma.value()
        self.params.Curr_RefSpeed = self.spin_curr.value()


class MooringDialog(QDialog):
    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self.setWindowTitle("Mooring Settings")
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.combo_model = QComboBox()
        self.combo_model.addItems(["Simple", "MAP++", "MoorDyn", "MoorEmm", "AI Mooring"])
        self.combo_model.setCurrentIndex(self.params.Moor_ModelType)
        form.addRow("Mooring Model:", self.combo_model)

        self.spin_cb = QDoubleSpinBox()
        self.spin_cb.setRange(0.0, 1.0)
        self.spin_cb.setSingleStep(0.05)
        self.spin_cb.setValue(self.params.Moor_CB)
        form.addRow("Seabed Friction Cb:", self.spin_cb)

        self.spin_length = QDoubleSpinBox()
        self.spin_length.setRange(100, 2000)
        self.spin_length.setValue(self.params.Moor_LineLength)
        self.spin_length.setSuffix(" m")
        form.addRow("Line Length:", self.spin_length)

        self.spin_mass = QDoubleSpinBox()
        self.spin_mass.setRange(1, 1000)
        self.spin_mass.setValue(self.params.Moor_LineMass)
        self.spin_mass.setSuffix(" kg/m")
        form.addRow("Line Mass:", self.spin_mass)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply(self):
        self.params.Moor_ModelType = self.combo_model.currentIndex()
        self.params.Moor_CB = self.spin_cb.value()
        self.params.Moor_LineLength = self.spin_length.value()
        self.params.Moor_LineMass = self.spin_mass.value()


class StructureDialog(QDialog):
    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self.setWindowTitle("Turbine Structure Settings")
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(10, 200)
        self.spin_radius.setValue(self.params.R)
        self.spin_radius.setSuffix(" m")
        form.addRow("Rotor Radius:", self.spin_radius)

        self.spin_hub = QDoubleSpinBox()
        self.spin_hub.setRange(10, 200)
        self.spin_hub.setValue(self.params.HubHeight)
        self.spin_hub.setSuffix(" m")
        form.addRow("Hub Height:", self.spin_hub)

        self.spin_tower = QDoubleSpinBox()
        self.spin_tower.setRange(10, 200)
        self.spin_tower.setValue(self.params.TowerHeight)
        self.spin_tower.setSuffix(" m")
        form.addRow("Tower Height:", self.spin_tower)

        self.spin_mass = QDoubleSpinBox()
        self.spin_mass.setRange(1e5, 1e8)
        self.spin_mass.setValue(self.params.Mass_Platform)
        self.spin_mass.setSuffix(" kg")
        self.spin_mass.setDecimals(0)
        form.addRow("Platform Mass:", self.spin_mass)

        self.spin_nac = QDoubleSpinBox()
        self.spin_nac.setRange(1e4, 1e7)
        self.spin_nac.setValue(self.params.Mass_Nacelle)
        self.spin_nac.setSuffix(" kg")
        self.spin_nac.setDecimals(0)
        form.addRow("Nacelle Mass:", self.spin_nac)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply(self):
        self.params.R = self.spin_radius.value()
        self.params.HubHeight = self.spin_hub.value()
        self.params.TowerHeight = self.spin_tower.value()
        self.params.Mass_Platform = self.spin_mass.value()
        self.params.Mass_Nacelle = self.spin_nac.value()
