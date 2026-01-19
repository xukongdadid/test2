from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QGridLayout, QLabel
from PySide6.QtCore import Qt


class TelemetryDockWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.lbls = {}

        grp_dof = QGroupBox("Platform DOF")
        grid_dof = QGridLayout()
        self._add_label(grid_dof, 0, 0, "Surge", "dof_Surge")
        self._add_label(grid_dof, 1, 0, "Sway", "dof_Sway")
        self._add_label(grid_dof, 2, 0, "Heave", "dof_Heave")
        self._add_label(grid_dof, 0, 2, "Roll", "dof_Roll")
        self._add_label(grid_dof, 1, 2, "Pitch", "dof_Pitch")
        self._add_label(grid_dof, 2, 2, "Yaw", "dof_Yaw")
        grp_dof.setLayout(grid_dof)
        layout.addWidget(grp_dof)

        grp_env = QGroupBox("Environment")
        grid_env = QGridLayout()
        self._add_label(grid_env, 0, 0, "Wind (m/s)", "env_wind_speed")
        self._add_label(grid_env, 1, 0, "Wave (m)", "env_wave_elev")
        self._add_label(grid_env, 0, 2, "Thrust (N)", "thrust")
        self._add_label(grid_env, 1, 2, "Gen Tq (Nm)", "gen_torque")
        grp_env.setLayout(grid_env)
        layout.addWidget(grp_env)

        grp_load = QGroupBox("Loads")
        grid_load = QGridLayout()
        self._add_label(grid_load, 0, 0, "Hydro Fx", "HydroFx")
        self._add_label(grid_load, 0, 2, "Moor Fx", "MoorFx")
        self._add_label(grid_load, 1, 0, "Hydro My", "HydroMy")
        self._add_label(grid_load, 1, 2, "Moor My", "MoorMy")
        grp_load.setLayout(grid_load)
        layout.addWidget(grp_load)

        layout.addStretch()

    def _add_label(self, grid, row, col, name, key):
        title = QLabel(f"{name}:")
        title.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        val = QLabel("--")
        val.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        val.setStyleSheet("font-weight: bold;")
        grid.addWidget(title, row, col)
        grid.addWidget(val, row, col + 1)
        self.lbls[key] = val

    def update_data(self, data):
        for key, lbl in self.lbls.items():
            if key in data:
                try:
                    lbl.setText(f"{float(data[key]):.3f}")
                except Exception:
                    lbl.setText(str(data[key]))
