from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QPushButton, QDoubleSpinBox, QGroupBox)


class StructureDialog(QDialog):
    def __init__(self, current_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Structure Parameters")
        self.resize(520, 480)
        self.params = current_params

        self.layout = QVBoxLayout(self)

        grp_geom = QGroupBox("Geometry")
        form_geom = QFormLayout()
        self.spin_r = QDoubleSpinBox(); self.spin_r.setRange(10, 200); self.spin_r.setSuffix(" m")
        self.spin_hub = QDoubleSpinBox(); self.spin_hub.setRange(10, 200); self.spin_hub.setSuffix(" m")
        self.spin_tower = QDoubleSpinBox(); self.spin_tower.setRange(10, 200); self.spin_tower.setSuffix(" m")
        form_geom.addRow("Rotor Radius:", self.spin_r)
        form_geom.addRow("Hub Height:", self.spin_hub)
        form_geom.addRow("Tower Height:", self.spin_tower)
        grp_geom.setLayout(form_geom)
        self.layout.addWidget(grp_geom)

        grp_mass = QGroupBox("Mass & Inertia")
        form_mass = QFormLayout()
        self.spin_m_blade = QDoubleSpinBox(); self.spin_m_blade.setRange(1, 1e7); self.spin_m_blade.setSuffix(" kg")
        self.spin_m_nac = QDoubleSpinBox(); self.spin_m_nac.setRange(1, 1e7); self.spin_m_nac.setSuffix(" kg")
        self.spin_m_hub = QDoubleSpinBox(); self.spin_m_hub.setRange(1, 1e7); self.spin_m_hub.setSuffix(" kg")
        self.spin_m_plat = QDoubleSpinBox(); self.spin_m_plat.setRange(1, 1e9); self.spin_m_plat.setSuffix(" kg")
        self.spin_m_ballast = QDoubleSpinBox(); self.spin_m_ballast.setRange(0, 1e9); self.spin_m_ballast.setSuffix(" kg")
        self.spin_i_roll = QDoubleSpinBox(); self.spin_i_roll.setRange(0, 1e12); self.spin_i_roll.setSuffix(" kg*m^2")
        self.spin_i_pitch = QDoubleSpinBox(); self.spin_i_pitch.setRange(0, 1e12); self.spin_i_pitch.setSuffix(" kg*m^2")
        self.spin_i_yaw = QDoubleSpinBox(); self.spin_i_yaw.setRange(0, 1e12); self.spin_i_yaw.setSuffix(" kg*m^2")
        form_mass.addRow("Blade Mass:", self.spin_m_blade)
        form_mass.addRow("Nacelle Mass:", self.spin_m_nac)
        form_mass.addRow("Hub Mass:", self.spin_m_hub)
        form_mass.addRow("Platform Mass:", self.spin_m_plat)
        form_mass.addRow("Ballast Mass:", self.spin_m_ballast)
        form_mass.addRow("I Roll:", self.spin_i_roll)
        form_mass.addRow("I Pitch:", self.spin_i_pitch)
        form_mass.addRow("I Yaw:", self.spin_i_yaw)
        grp_mass.setLayout(form_mass)
        self.layout.addWidget(grp_mass)

        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("Apply"); self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("Cancel"); self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_ok); btn_layout.addWidget(self.btn_cancel)
        self.layout.addLayout(btn_layout)

        self.load_params()

    def load_params(self):
        self.spin_r.setValue(self.params.R)
        self.spin_hub.setValue(self.params.HubHeight)
        self.spin_tower.setValue(self.params.TowerHeight)
        self.spin_m_blade.setValue(self.params.Mass_Blade)
        self.spin_m_nac.setValue(self.params.Mass_Nacelle)
        self.spin_m_hub.setValue(self.params.Mass_Hub)
        self.spin_m_plat.setValue(self.params.Mass_Platform)
        self.spin_m_ballast.setValue(self.params.Mass_Ballast)
        self.spin_i_roll.setValue(self.params.I_Roll)
        self.spin_i_pitch.setValue(self.params.I_Pitch)
        self.spin_i_yaw.setValue(self.params.I_Yaw)

    def get_params(self):
        self.params.R = self.spin_r.value()
        self.params.HubHeight = self.spin_hub.value()
        self.params.TowerHeight = self.spin_tower.value()
        self.params.Mass_Blade = self.spin_m_blade.value()
        self.params.Mass_Nacelle = self.spin_m_nac.value()
        self.params.Mass_Hub = self.spin_m_hub.value()
        self.params.Mass_Platform = self.spin_m_plat.value()
        self.params.Mass_Ballast = self.spin_m_ballast.value()
        self.params.I_Roll = self.spin_i_roll.value()
        self.params.I_Pitch = self.spin_i_pitch.value()
        self.params.I_Yaw = self.spin_i_yaw.value()
        self.params.Mtop = self.params.Mass_Nacelle + self.params.Mass_Hub + 3 * self.params.Mass_Blade
        return self.params
