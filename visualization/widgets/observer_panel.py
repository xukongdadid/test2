from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSpinBox, QLabel, QScrollArea
from visualization.widgets.observer_scope import TimeScopeWidget, FreqScopeWidget


class ObserverPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.spin_time_scopes = QSpinBox()
        self.spin_time_scopes.setRange(1, 6)
        self.spin_time_scopes.setValue(1)
        self.spin_freq_scopes = QSpinBox()
        self.spin_freq_scopes.setRange(0, 6)
        self.spin_freq_scopes.setValue(0)
        self.btn_apply = QPushButton("Apply Scopes")
        controls.addWidget(QLabel("Time Scopes:"))
        controls.addWidget(self.spin_time_scopes)
        controls.addWidget(QLabel("Freq Scopes:"))
        controls.addWidget(self.spin_freq_scopes)
        controls.addWidget(self.btn_apply)
        controls.addStretch()
        layout.addLayout(controls)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll.setWidget(self.scroll_content)
        layout.addWidget(self.scroll)

        self.time_scopes = []
        self.freq_scopes = []
        self.btn_apply.clicked.connect(self.rebuild_scopes)

        self.rebuild_scopes()

    def rebuild_scopes(self):
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
        self.time_scopes = []
        self.freq_scopes = []

        for _ in range(self.spin_time_scopes.value()):
            scope = TimeScopeWidget()
            self.scroll_layout.addWidget(scope)
            self.time_scopes.append(scope)

        for _ in range(self.spin_freq_scopes.value()):
            scope = FreqScopeWidget()
            self.scroll_layout.addWidget(scope)
            self.freq_scopes.append(scope)

        self.scroll_layout.addStretch()

    def set_available_channels(self, channels: list[str]) -> None:
        for scope in self.time_scopes + self.freq_scopes:
            scope.set_available_channels(channels)

    def update_data(self, data_frame: dict) -> None:
        for scope in self.time_scopes + self.freq_scopes:
            scope.update_data(data_frame)
