import sys
import os
from PySide6.QtWidgets import QApplication
from visualization.main_window import MainWindow

VERSION = "0.0.1"
APP_NAME = "LabFAST"

def main():
    print(f"Starting {APP_NAME} V{VERSION}...")
    
    os.environ["QT_API"] = "pyside6"
    if hasattr(sys, 'frozen'):
        os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName(APP_NAME)
    
    window = MainWindow()
    window.setWindowTitle(f"{APP_NAME} V{VERSION}")
    window.show()
    
    print("System Initialized. Ready for Simulation.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()