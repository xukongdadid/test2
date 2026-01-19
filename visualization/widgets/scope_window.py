import csv
import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                               QFileDialog, QTreeWidget, QTreeWidgetItem, QDialogButtonBox,
                               QCheckBox)
from PySide6.QtCore import Qt

from visualization.widgets.dock_scope import ScopeDockWidget
from visualization.widgets.dock_control import CHANNEL_CONFIG


COLOR_CYCLE = [
    "#ff6b6b", "#4ecdc4", "#ffd93d", "#5f6caf",
    "#ff9f1c", "#6bcb77", "#4d96ff", "#845ec2",
]


class ChannelSelectDialog(QDialog):
    def __init__(self, selected_keys=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Channels")
        self.resize(420, 520)
        self.selected_keys = set(selected_keys or [])
        self.layout = QVBoxLayout(self)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Channel"])
        self.layout.addWidget(self.tree)
        self._populate_tree()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        self.layout.addWidget(btns)

    def _populate_tree(self):
        self.tree.clear()
        for group, items in CHANNEL_CONFIG.items():
            g_item = QTreeWidgetItem(self.tree)
            g_item.setText(0, group)
            g_item.setFlags(Qt.ItemIsEnabled)
            for name, key in items.items():
                c_item = QTreeWidgetItem(g_item)
                c_item.setText(0, name)
                c_item.setData(0, Qt.UserRole, key)
                c_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                c_item.setCheckState(0, Qt.Checked if key in self.selected_keys else Qt.Unchecked)
        self.tree.expandAll()

    def get_channels(self):
        channels = []
        root = self.tree.invisibleRootItem()
        idx = 0
        for i in range(root.childCount()):
            group = root.child(i)
            for j in range(group.childCount()):
                item = group.child(j)
                if item.checkState(0) == Qt.Checked:
                    key = item.data(0, Qt.UserRole)
                    color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
                    idx += 1
                    channels.append({
                        "key": key,
                        "name": item.text(0),
                        "color": color,
                        "style": "Solid",
                    })
        return channels


class TimeScopeWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Time Scope")
        self.resize(900, 600)
        self.setWindowFlag(Qt.Window, True)

        self.scope = ScopeDockWidget(enable_recording=False)
        self.channels = []

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_channels = QPushButton("Channels")
        self.btn_channels.clicked.connect(self.configure_channels)
        self.btn_import = QPushButton("Import History")
        self.btn_import.clicked.connect(self.import_history)
        top.addWidget(self.btn_channels)
        top.addWidget(self.btn_import)
        top.addStretch()
        layout.addLayout(top)
        layout.addWidget(self.scope)

    def configure_channels(self):
        dlg = ChannelSelectDialog([c["key"] for c in self.channels], self)
        if dlg.exec():
            self.channels = dlg.get_channels()
            self.scope.overlays = []
            self.scope.update_channels(self.channels)

    def update_data(self, data):
        self.scope.update_data(data)

    def reset_session(self):
        self.scope.reset_session()

    def import_history(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Import History", "", "CSV (*.csv)")
        if not fn:
            return
        try:
            with open(fn, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            return
        if not rows:
            return

        times = []
        data_by_key = {c["key"]: [] for c in self.channels}
        for row in rows:
            t = row.get("time") or row.get("Time")
            if t is None:
                continue
            times.append(float(t))
            for conf in self.channels:
                key = conf["key"]
                val = row.get(key)
                if val is not None:
                    data_by_key[key].append(float(val))

        for conf in self.channels:
            key = conf["key"]
            if key in data_by_key and len(data_by_key[key]) == len(times):
                self.scope.add_overlay(key, conf["name"], times, data_by_key[key], conf["color"])


class FrequencyScopeWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Frequency Scope")
        self.resize(900, 600)
        self.setWindowFlag(Qt.Window, True)

        self.channels = []
        self.time_history = []
        self.data_buffers = {}
        self.curves = {}
        self.overlays = []
        self.auto_refresh = True

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_channels = QPushButton("Channels")
        self.btn_channels.clicked.connect(self.configure_channels)
        self.chk_auto = QCheckBox("Auto Refresh")
        self.chk_auto.setChecked(True)
        self.chk_auto.toggled.connect(self._set_auto_refresh)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_fft)
        self.btn_import = QPushButton("Import History")
        self.btn_import.clicked.connect(self.import_history)
        top.addWidget(self.btn_channels)
        top.addWidget(self.chk_auto)
        top.addWidget(self.btn_refresh)
        top.addWidget(self.btn_import)
        top.addStretch()
        layout.addLayout(top)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("k")
        self.plot.showGrid(x=True, y=True)
        self.plot.addLegend()
        layout.addWidget(self.plot)

    def _set_auto_refresh(self, checked):
        self.auto_refresh = checked

    def configure_channels(self):
        dlg = ChannelSelectDialog([c["key"] for c in self.channels], self)
        if dlg.exec():
            self.channels = dlg.get_channels()
            self.overlays = []
            self._reset_plot()

    def _reset_plot(self):
        self.plot.clear()
        self.curves = {}
        for conf in self.channels:
            pen = pg.mkPen(color=conf["color"], width=2)
            curve = self.plot.plot(pen=pen, name=conf["name"])
            self.curves[conf["key"]] = curve
        self._redraw_overlays()

    def update_data(self, data):
        t = data.get("time", data.get("Time", None))
        if t is None:
            return
        self.time_history.append(float(t))
        for conf in self.channels:
            key = conf["key"]
            self.data_buffers.setdefault(key, []).append(float(data.get(key, 0.0)))
        if self.auto_refresh and len(self.time_history) % 20 == 0:
            self.refresh_fft()

    def reset_session(self):
        self.time_history = []
        self.data_buffers = {}
        self.overlays = []
        self._reset_plot()

    def refresh_fft(self):
        if len(self.time_history) < 4:
            return
        dt = np.mean(np.diff(self.time_history))
        if dt <= 0:
            return
        for conf in self.channels:
            key = conf["key"]
            values = self.data_buffers.get(key, [])
            if len(values) < 4:
                continue
            y = np.array(values) - np.mean(values)
            freq = np.fft.rfftfreq(len(y), dt)
            amp = np.abs(np.fft.rfft(y)) / max(len(y), 1)
            curve = self.curves.get(key)
            if curve:
                curve.setData(freq, amp)

    def import_history(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Import History", "", "CSV (*.csv)")
        if not fn:
            return
        try:
            with open(fn, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            return
        if not rows:
            return
        times = []
        data_by_key = {c["key"]: [] for c in self.channels}
        for row in rows:
            t = row.get("time") or row.get("Time")
            if t is None:
                continue
            times.append(float(t))
            for conf in self.channels:
                key = conf["key"]
                val = row.get(key)
                if val is not None:
                    data_by_key[key].append(float(val))
        if len(times) < 4:
            return
        dt = np.mean(np.diff(times))
        if dt <= 0:
            return
        for conf in self.channels:
            key = conf["key"]
            values = data_by_key.get(key, [])
            if len(values) < 4:
                continue
            y = np.array(values) - np.mean(values)
            freq = np.fft.rfftfreq(len(y), dt)
            amp = np.abs(np.fft.rfft(y)) / max(len(y), 1)
            self._add_overlay(conf["name"], freq, amp, conf["color"])

    def _add_overlay(self, name, freq, amp, color):
        pen = pg.mkPen(color=color, style=Qt.DashLine, width=1.6)
        label = f"{name} (history)"
        self.plot.plot(freq, amp, pen=pen, name=label)
        self.overlays.append({"name": name, "freq": freq, "amp": amp, "color": color})

    def _redraw_overlays(self):
        for overlay in self.overlays:
            pen = pg.mkPen(color=overlay["color"], style=Qt.DashLine, width=1.6)
            label = f'{overlay["name"]} (history)'
            self.plot.plot(overlay["freq"], overlay["amp"], pen=pen, name=label)
