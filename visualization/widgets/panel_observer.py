from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QFormLayout,
                               QSpinBox, QPushButton, QLabel, QHBoxLayout)
from PySide6.QtCore import Signal


class ObserverPanel(QWidget):
    sig_create_time_scopes = Signal(int)
    sig_create_freq_scopes = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        title = QLabel("Observer Mode")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        grp_time = QGroupBox("Time-Domain Scopes")
        form_time = QFormLayout()
        self.spin_time_count = QSpinBox(); self.spin_time_count.setRange(1, 8); self.spin_time_count.setValue(1)
        btn_time = QPushButton("Create")
        btn_time.clicked.connect(self._create_time_scopes)
        row_time = QHBoxLayout()
        row_time.addWidget(self.spin_time_count)
        row_time.addWidget(btn_time)
        form_time.addRow("Count:", row_time)
        grp_time.setLayout(form_time)
        layout.addWidget(grp_time)

        grp_freq = QGroupBox("Frequency-Domain Scopes")
        form_freq = QFormLayout()
        self.spin_freq_count = QSpinBox(); self.spin_freq_count.setRange(1, 8); self.spin_freq_count.setValue(1)
        btn_freq = QPushButton("Create")
        btn_freq.clicked.connect(self._create_freq_scopes)
        row_freq = QHBoxLayout()
        row_freq.addWidget(self.spin_freq_count)
        row_freq.addWidget(btn_freq)
        form_freq.addRow("Count:", row_freq)
        grp_freq.setLayout(form_freq)
        layout.addWidget(grp_freq)

        hint = QLabel("Tip: Each scope lets you select channels and compare history data.")
        hint.setStyleSheet("color: #6b7280;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch()

    def _create_time_scopes(self):
        self.sig_create_time_scopes.emit(self.spin_time_count.value())

    def _create_freq_scopes(self):
        self.sig_create_freq_scopes.emit(self.spin_freq_count.value())
