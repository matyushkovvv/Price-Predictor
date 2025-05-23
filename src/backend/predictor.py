import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler

class StockPredictorBackend:
    def __init__(self):
        self.data = None
        self.models = {}
        self.target_column = 'close'
    
    def load_data_from_yfinance(self, ticker, period='1y'):
        """Загрузка данных по тикеру из Yahoo Finance"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365) if period == '1y' else end_date - timedelta(days=365*5)
            
            data = yf.download(ticker, period=period, interval='1d')
            
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
        data['ema30'] = close.ewm(span=30, adjust=False).mean().round(2)

        data = data.dropna()
        
        return data
    
    def get_data_info(self):
        if self.data is None:
            return "Данные не загружены"
        
        info = [
            f"Количество строк: {len(self.data)}",
            f"Количество столбцев: {len(self.data.columns)}",

        ]
        return "\n".join(info)
    
    def get_plot_data(self):
        """Возвращает данные для графика с корректными датами"""
        if self.data is None or 'close' not in self.data.columns:
            return None
        
        if 'date' in self.data.columns:
            try:
                dates = pd.to_datetime(self.data['date'])
                return pd.Series(self.data['close'].values, index=dates)
            except:
                pass
        
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
        start_time = time.time()
        if self.data is None:
            return False, "Сначала загрузите данные!"

        try:
            df = self.data.copy()

            if 'close' not in df.columns:
                return False, "Столбец 'close' не найден"

            # 1. Создаем целевую переменную — цену следующего дня
            df['target'] = df['close'].shift(-1)
            df = df.dropna()

            # 2. Исключаем ненужные признаки
            exclude_cols = ['Date', 'close', 'target']
            X = df.drop(columns=exclude_cols)
            y = df['target']

            if len(df) < 10:
                return False, "Недостаточно данных для обучения"

            # 3. Проверка на пропущенные значения
            if X.isnull().any().any() or y.isnull().any():
                return False, "Обнаружены пропущенные значения"

            # 4. Хронологическое разбиение
            train_size = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

            # 5. Масштабирование
            scaler = RobustScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            self.scaler = scaler  # сохраняем на будущее

            # 6. Выбор модели
            if model_name == "Линейная регрессия":
                model = LinearRegression()
            elif model_name == "Случайный лес":
                model = RandomForestRegressor(
                    n_estimators=150, max_depth=7, min_samples_leaf=3, random_state=42, n_jobs=-1
                )
            elif model_name == "Градиентный бустинг":
                model = GradientBoostingRegressor(
                    n_estimators=120, learning_rate=0.08, max_depth=4,
                    min_samples_leaf=3, random_state=42
                )
            elif model_name == "Нейронная сеть":
                model = MLPRegressor(
                    hidden_layer_sizes=(64, 32), max_iter=800,
                    early_stopping=True, random_state=42, learning_rate_init=0.2
                )
            else:
                return False, "Неизвестная модель"

            # 7. Обучение и предсказание
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            learn_time = time.time() - start_time

            # 8. Сохраняем модель
            self.models[model_name] = model

            return True, {
                'message': f"Модель {model_name} обучена\nMAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}\nВремя: {learn_time:.2f} сек",
                'y_test': y_test.values,
                'y_pred': y_pred,
                'test_dates': df['date'].iloc[train_size:].values if 'date' in df.columns else None,
                'features': X.columns.tolist()
            }

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
        

    