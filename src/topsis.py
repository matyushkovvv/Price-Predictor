import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# Метрики моделей
data = pd.DataFrame({
    'MAE': [4.81, 11.88, 11.92, 8.22],
    'MSE': [48.8, 233.16, 243.58, 117.23],
    'R2':  [0.89, 0.49, 0.46, 0.74],
}, index=['LinearRegression', 'RandomForest', 'GBR', 'MLP'])

# 1. Векторная нормализация
norm_data = pd.DataFrame(normalize(data, axis=0), columns=data.columns, index=data.index)

# 2. Расчёт весов по CRITIC
std = norm_data.std()
corr_matrix = norm_data.corr()
conflict = 1 - corr_matrix.abs().sum()
critic_score = std * conflict
weights = critic_score / critic_score.sum()

# 3. Заданные идеальная и анти-идеальная точки
ideal_point = np.array([0, 0, 1])   # MAE, MSE, R2
anti_ideal_point = np.array([1, 1, 0])

# 4. Расстояния до идеальной и анти-идеальной точек
weighted_data = norm_data * weights.values

D_plus = np.linalg.norm(weighted_data.values - ideal_point * weights.values, axis=1)
D_minus = np.linalg.norm(weighted_data.values - anti_ideal_point * weights.values, axis=1)

# 5. TOPSIS score
scores = D_minus / (D_plus + D_minus)

# 6. Финальный результат
result = pd.DataFrame({
    'TOPSIS Score': scores,
    'D+': D_plus,
    'D-': D_minus
}, index=data.index).sort_values(by='TOPSIS Score', ascending=False)

print(result)

# 7. Визуализация 3D
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Координаты моделей
x, y, z = weighted_data['MAE'], weighted_data['MSE'], weighted_data['R2']
ax.scatter(x, y, z, c='blue', label='Models')
for i, name in enumerate(weighted_data.index):
    ax.text(x[i], y[i], z[i], name, fontsize=9)

# Идеальная и анти-идеальная точки
ax.scatter(*ideal_point, c='green', s=100, label='Ideal')
ax.scatter(*anti_ideal_point, c='red', s=100, label='Anti-Ideal')

ax.set_xlabel('MAE (норм.)')
ax.set_ylabel('MSE (норм.)')
ax.set_zlabel('R² (норм.)')
ax.set_title('TOPSIS + CRITIC: Визуализация моделей')
ax.legend()
plt.tight_layout()
plt.show()
