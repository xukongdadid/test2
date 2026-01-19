from PySide6.QtWidgets import QMenu
from PySide6.QtCore import Signal, QPoint, Qt
import pyqtgraph.opengl as gl


class InteractiveGLViewWidget(gl.GLViewWidget):
    clicked_sky = Signal()
    clicked_wave = Signal()
    clicked_mooring = Signal()
    clicked_structure = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._context_menu = QMenu(self)
        self._context_menu.addAction("Edit Wind (Sky)", self.clicked_sky.emit)
        self._context_menu.addAction("Edit Wave/Current", self.clicked_wave.emit)
        self._context_menu.addAction("Edit Mooring", self.clicked_mooring.emit)
        self._context_menu.addAction("Edit Structure", self.clicked_structure.emit)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._context_menu.exec(event.globalPosition().toPoint())
            return

        if event.button() == Qt.LeftButton:
            self._emit_click_by_region(event.position().toPoint())
        super().mousePressEvent(event)

    def _emit_click_by_region(self, pos: QPoint) -> None:
        if self.height() == 0 or self.width() == 0:
            return

        y_ratio = pos.y() / self.height()
        x_ratio = pos.x() / self.width()

        if y_ratio < 0.25:
            self.clicked_sky.emit()
        elif y_ratio > 0.75:
            self.clicked_wave.emit()
        elif x_ratio < 0.25:
            self.clicked_mooring.emit()
        else:
            self.clicked_structure.emit()
