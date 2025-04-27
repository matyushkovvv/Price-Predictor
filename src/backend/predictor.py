import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

class StockPredictorBackend:
    def __init__(self):
        self.data = None
        self.models = {}
        self.scaler = MinMaxScaler()
        self.target_column = 'close'
    
    def load_data_from_yfinance(self, ticker, period='1y'):
        """Загрузка данных по тикеру из Yahoo Finance"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365) if period == '1y' else end_date - timedelta(days=365*5)
            
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if data.empty:
                return False, f"Не удалось получить данные для тикера {ticker}"
            
            # Обработка MultiIndex колонок
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0).str.lower()
            
            # Сбрасываем индекс и переименовываем
            data = data.reset_index()
            data = data.rename(columns={'index': 'date'})
            
            # Добавляем технические индикаторы
            data = self.add_technical_indicators(data)
            
            # Округляем значения
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].round(2)
            
            self.data = data
            return True, f"Данные для {ticker} успешно загружены"
        except Exception as e:
            return False, f"Ошибка при загрузке данных: {str(e)}"
    
    def add_technical_indicators(self, data):
        """Добавление технических индикаторов"""
        close = data['close']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        data['rsi'] = (100 - (100 / (1 + rs))).round(2)
        
        # EMA
        data['ema10'] = close.ewm(span=10, adjust=False).mean().round(2)
        data['ema20'] = close.ewm(span=20, adjust=False).mean().round(2)
        
        return data
    
    def get_data_info(self):
        if self.data is None:
            return "Данные не загружены"
        
        info = [
            f"Количество строк: {len(self.data)}",
            f"Количество столбцев: {len(self.data.columns)}",
            "\nПервые 5 строк:",
            str(self.data.head()),
            "\nОписание данных:",
            str(self.data.describe())
        ]
        return "\n".join(info)
    
    def get_plot_data(self):
        """Возвращает данные для графика"""
        if self.data is None or 'close' not in self.data.columns:
            return None
        
        if 'date' in self.data.columns:
            return pd.Series(
                self.data['close'].values,
                index=pd.to_datetime(self.data['date'])
            )
        return pd.Series(self.data['close'].values)
    
    def get_table_data(self):
        return self.data
    
    def get_features(self):
        if self.data is None or 'close' not in self.data.columns:
            return []
        
        exclude = ['date', 'close']
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        return [col for col in numeric_cols if col not in exclude]
    
    def train_model(self, model_name, lookback=5):
        if self.data is None:
            return False, "Сначала загрузите данные!"
        
        try:
            # Создаем лаговые переменные
            processed_data = self.data.copy()
            for i in range(1, lookback + 1):
                processed_data[f'close_lag_{i}'] = processed_data['close'].shift(i)
            
            processed_data = processed_data.dropna()
            
            if len(processed_data) < 10:
                return False, "Недостаточно данных для обучения"
            
            # Подготовка данных
            X = processed_data.drop(['date', 'close'], axis=1, errors='ignore')
            y = processed_data['close']
            
            X_scaled = self.scaler.fit_transform(X)
            
            # Разделение на train/test (без перемешивания)
            train_size = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Обучение модели
            if model_name == "Линейная регрессия":
                model = LinearRegression()
            elif model_name == "Случайный лес":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_name == "Градиентный бустинг":
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif model_name == "Нейронная сеть":
                model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            else:
                return False, "Неизвестная модель"
            
            model.fit(X_train, y_train)
            self.models[model_name] = model
            
            # Оценка модели
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return True, (
                f"Модель {model_name} обучена\nMSE: {mse:.2f}, R2: {r2:.2f}",
                (y_test, y_pred)
            )
        except Exception as e:
            return False, f"Ошибка при обучении: {str(e)}"
    
    def make_prediction(self, model_name, feature_values):
        if not self.models:
            return False, "Сначала обучите модель!"
        
        model = self.models.get(model_name)
        if model is None:
            return False, f"Модель {model_name} не обучена"
        
        try:
            input_data = np.array(feature_values).reshape(1, -1)
            input_data_scaled = self.scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)
            return True, f"Прогноз: {prediction[0]:.2f}"
        except Exception as e:
            return False, f"Ошибка прогноза: {str(e)}"