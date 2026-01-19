from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout, QGroupBox
from PySide6.QtCore import Qt


class LiveDataPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        grp = QGroupBox("Live Metrics")
        grid = QGridLayout()
        grp.setLayout(grid)
        layout.addWidget(grp)
        layout.addStretch()

        self.labels = {}
        fields = [
            ("Wind Speed", "env_wind_speed", "m/s"),
            ("Wave Elevation", "env_wave_elev", "m"),
            ("Current Speed", "env_current_speed", "m/s"),
            ("Power", "GenPwr", "MW"),
            ("Surge", "dof_Surge", "m"),
            ("Pitch", "dof_Pitch", "deg"),
            ("Mooring Fx", "MoorFx", "N"),
            ("Hydro Fx", "HydroFx", "N"),
        ]

        for row, (name, key, unit) in enumerate(fields):
            label_name = QLabel(f"{name}:")
            label_value = QLabel("--")
            label_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(label_name, row, 0)
            grid.addWidget(label_value, row, 1)
            grid.addWidget(QLabel(unit), row, 2)
            self.labels[key] = label_value

    def update_values(self, data: dict) -> None:
        for key, label in self.labels.items():
            if key not in data:
                continue
            value = data[key]
            if key == "GenPwr":
                value = value / 1e6
            if isinstance(value, float):
                label.setText(f"{value:.3f}")
            else:
                label.setText(str(value))
