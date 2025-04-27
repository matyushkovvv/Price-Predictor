from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
                             QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
                             QTabWidget, QGroupBox, QFormLayout)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class StockPredictorFrontend(QMainWindow):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.setWindowTitle("Прогнозирование котировок")
        self.setGeometry(100, 100, 1200, 800)
        self.init_ui()
    
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Левая панель
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Вкладки
        self.tabs = QTabWidget()
        
        # Вкладка загрузки по тикеру
        ticker_tab = QWidget()
        ticker_layout = QVBoxLayout(ticker_tab)
        
        ticker_group = QGroupBox("Параметры загрузки")
        form_layout = QFormLayout()
        
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("AAPL, MSFT...")
        form_layout.addRow("Тикер:", self.ticker_input)
        
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1 год", "5 лет"])
        form_layout.addRow("Период:", self.period_combo)
        
        ticker_group.setLayout(form_layout)
        ticker_layout.addWidget(ticker_group)
        
        self.load_ticker_btn = QPushButton("Загрузить данные")
        self.load_ticker_btn.clicked.connect(self.load_data_from_ticker)
        ticker_layout.addWidget(self.load_ticker_btn)
        
        self.tabs.addTab(ticker_tab, "Yahoo Finance")
        
        left_layout.addWidget(self.tabs)
        
        # Настройки модели
        model_group = QGroupBox("Настройки модели")
        model_layout = QVBoxLayout()
        
        model_layout.addWidget(QLabel("Модель:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Линейная регрессия", "Случайный лес", 
                                    "Градиентный бустинг", "Нейронная сеть"])
        model_layout.addWidget(self.model_selector)
        
        self.train_btn = QPushButton("Обучить модель")
        self.train_btn.clicked.connect(self.train_model)
        model_layout.addWidget(self.train_btn)
        
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)
        
        # Прогноз
        predict_group = QGroupBox("Прогноз")
        predict_layout = QVBoxLayout()
        
        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(2)
        self.feature_table.setHorizontalHeaderLabels(["Признак", "Значение"])
        self.feature_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        predict_layout.addWidget(self.feature_table)
        
        self.predict_btn = QPushButton("Сделать прогноз")
        self.predict_btn.clicked.connect(self.make_prediction)
        predict_layout.addWidget(self.predict_btn)
        
        predict_group.setLayout(predict_layout)
        left_layout.addWidget(predict_group)
        
        # Информация
        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        left_layout.addWidget(QLabel("Информация:"))
        left_layout.addWidget(self.info_display)
        
        # Правая панель
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # График
        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        
        # Таблица данных
        self.data_table = QTableWidget()
        right_layout.addWidget(self.data_table)
        
        # Компоновка
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)
        
        self.setCentralWidget(main_widget)
    
    def load_data_from_ticker(self):
        ticker = self.ticker_input.text().strip()
        if not ticker:
            self.info_display.setText("Введите тикер")
            return
        
        period = '1y' if self.period_combo.currentText() == "1 год" else '5y'
        success, message = self.backend.load_data_from_yfinance(ticker, period)
        
        self.info_display.setText(message)
        if success:
            self.update_ui()
    
    def update_ui(self):
        self.info_display.setText(self.backend.get_data_info())
        self.update_feature_table()
        self.show_data_table()
        self.plot_data()
    
    def update_feature_table(self):
        features = self.backend.get_features()
        self.feature_table.setRowCount(len(features))
        
        for i, feature in enumerate(features):
            self.feature_table.setItem(i, 0, QTableWidgetItem(feature))
            self.feature_table.setItem(i, 1, QTableWidgetItem("0"))
    
    def show_data_table(self):
        data = self.backend.get_table_data()
        if data is None:
            return
        
        self.data_table.setRowCount(len(data))
        self.data_table.setColumnCount(len(data.columns))
        self.data_table.setHorizontalHeaderLabels(data.columns)
        
        for i in range(len(data)):
            for j in range(len(data.columns)):
                value = data.iloc[i, j]
                display_value = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                self.data_table.setItem(i, j, QTableWidgetItem(display_value))
    
    def plot_data(self, y_test=None, y_pred=None):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        historical_data = self.backend.get_plot_data()
        if historical_data is not None:
            # Исторические данные
            ax.plot(historical_data.index, historical_data.values, 
                   label='Исторические данные', color='blue')
            
            # Тестовые и прогнозные данные
            if y_test is not None and y_pred is not None:
                test_dates = historical_data.index[-len(y_test):]
                ax.plot(test_dates, y_test.values, 
                       label='Фактические значения', color='green')
                ax.plot(test_dates, y_pred, 
                       label='Прогноз', color='red', linestyle='--')
            
            ax.set_title('Исторические данные и прогноз')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Цена закрытия')
            ax.legend()
            ax.grid(True)
            ax.tick_params(axis='x', rotation=45)
            self.figure.tight_layout()
            self.canvas.draw()
    
    def train_model(self):
        model_name = self.model_selector.currentText()
        success, result = self.backend.train_model(model_name)
        
        if success:
            message, (y_test, y_pred) = result
            self.info_display.setText(message)
            self.plot_data(y_test, y_pred)
        else:
            self.info_display.setText(result)
    
    def make_prediction(self):
        model_name = self.model_selector.currentText()
        features = self.backend.get_features()
        feature_values = []
        
        for i in range(len(features)):
            value_item = self.feature_table.item(i, 1)
            if value_item:
                try:
                    feature_values.append(float(value_item.text()))
                except ValueError:
                    self.info_display.setText(f"Некорректное значение для {features[i]}")
                    return
        
        success, message = self.backend.make_prediction(model_name, feature_values)
        self.info_display.setText(message)
