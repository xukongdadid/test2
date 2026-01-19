import sys
import os
from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from visualization.main_window import MainWindow

VERSION = "0.0.1"
APP_NAME = "LabFAST"

def main():
    print(f"Starting {APP_NAME} V{VERSION}...")
    
    os.environ["QT_API"] = "pyside6"
    if hasattr(sys, 'frozen'):
        os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']

    # Force desktop OpenGL (avoid ANGLE/ES fallback) for pyqtgraph.opengl
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseDesktopOpenGL)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QtGui.QSurfaceFormat.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName(APP_NAME)
    app.setFont(QFont("Segoe UI", 10))
    
    window = MainWindow()
    window.setWindowTitle(f"{APP_NAME} V{VERSION}")
    window.show()
    
    print("System Initialized. Ready for Simulation.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
