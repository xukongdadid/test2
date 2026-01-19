from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QDoubleSpinBox, QPushButton, QGroupBox)


class TurbineDialog(QDialog):
    def __init__(self, current_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Turbine Structure Parameters")
        self.resize(480, 420)
        self.params = current_params

        self.layout = QVBoxLayout(self)

        grp_geom = QGroupBox("Geometry")
        form_geom = QFormLayout()
        self.spin_radius = QDoubleSpinBox(); self.spin_radius.setRange(10, 200); self.spin_radius.setSuffix(" m")
        self.spin_hub = QDoubleSpinBox(); self.spin_hub.setRange(20, 200); self.spin_hub.setSuffix(" m")
        self.spin_tower = QDoubleSpinBox(); self.spin_tower.setRange(20, 200); self.spin_tower.setSuffix(" m")
        form_geom.addRow("Rotor Radius:", self.spin_radius)
        form_geom.addRow("Hub Height:", self.spin_hub)
        form_geom.addRow("Tower Height:", self.spin_tower)
        grp_geom.setLayout(form_geom)
        self.layout.addWidget(grp_geom)

        grp_mass = QGroupBox("Mass Properties")
        form_mass = QFormLayout()
        self.spin_blade = QDoubleSpinBox(); self.spin_blade.setRange(1e3, 1e6); self.spin_blade.setSuffix(" kg")
        self.spin_nac = QDoubleSpinBox(); self.spin_nac.setRange(1e3, 1e7); self.spin_nac.setSuffix(" kg")
        self.spin_hub_mass = QDoubleSpinBox(); self.spin_hub_mass.setRange(1e3, 1e7); self.spin_hub_mass.setSuffix(" kg")
        self.spin_plat = QDoubleSpinBox(); self.spin_plat.setRange(1e4, 1e9); self.spin_plat.setSuffix(" kg")
        self.spin_ballast = QDoubleSpinBox(); self.spin_ballast.setRange(0, 1e9); self.spin_ballast.setSuffix(" kg")
        form_mass.addRow("Blade Mass (each):", self.spin_blade)
        form_mass.addRow("Nacelle Mass:", self.spin_nac)
        form_mass.addRow("Hub Mass:", self.spin_hub_mass)
        form_mass.addRow("Platform Mass:", self.spin_plat)
        form_mass.addRow("Ballast Mass:", self.spin_ballast)
        grp_mass.setLayout(form_mass)
        self.layout.addWidget(grp_mass)

        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("Apply"); btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel"); btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_ok); btn_layout.addWidget(btn_cancel)
        self.layout.addLayout(btn_layout)

        self.load_params()

    def load_params(self):
        self.spin_radius.setValue(self.params.R)
        self.spin_hub.setValue(self.params.HubHeight)
        self.spin_tower.setValue(self.params.TowerHeight)
        self.spin_blade.setValue(self.params.Mass_Blade)
        self.spin_nac.setValue(self.params.Mass_Nacelle)
        self.spin_hub_mass.setValue(self.params.Mass_Hub)
        self.spin_plat.setValue(self.params.Mass_Platform)
        self.spin_ballast.setValue(getattr(self.params, "Mass_Ballast", 0.0))

    def get_params(self):
        self.params.R = self.spin_radius.value()
        self.params.HubHeight = self.spin_hub.value()
        self.params.TowerHeight = self.spin_tower.value()
        self.params.Mass_Blade = self.spin_blade.value()
        self.params.Mass_Nacelle = self.spin_nac.value()
        self.params.Mass_Hub = self.spin_hub_mass.value()
        self.params.Mass_Platform = self.spin_plat.value()
        self.params.Mass_Ballast = self.spin_ballast.value()
        self.params.Mtop = self.params.Mass_Nacelle + self.params.Mass_Hub + 3 * self.params.Mass_Blade
        return self.params
