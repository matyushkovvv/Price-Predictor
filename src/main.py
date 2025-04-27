import sys
from backend.predictor import StockPredictorBackend
from frontend.app import StockPredictorFrontend
from PyQt5.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    backend = StockPredictorBackend()
    window = StockPredictorFrontend(backend)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()