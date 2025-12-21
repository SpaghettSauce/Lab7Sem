import numpy as np
import matplotlib.pyplot as plt

# Исходные данные
v0 = 2.0                     # начальная скорость, м/с
alpha = np.log(2) / 4        # коэффициент сопротивления

# Временной интервал
t = np.linspace(0, 20, 500)

# Закон изменения скорости
v = v0 * np.exp(-alpha * t)

# Закон изменения координаты
x = (v0 / alpha) * (1 - np.exp(-alpha * t))

# Время достижения скорости 0.25 м/с
v_target = 0.25
t_target = np.log(v0 / v_target) / alpha

# Полный путь до остановки
S_total = v0 / alpha

# Вывод результатов
print(f"Коэффициент сопротивления alpha = {alpha:.4f}")
print(f"Время достижения скорости 0.25 м/с: t = {t_target:.2f} с")
print(f"Полный путь до остановки: S = {S_total:.2f} м")

# Построение графиков
plt.figure()
plt.plot(t, v)
plt.xlabel("Время, с")
plt.ylabel("Скорость, м/с")
plt.title("Зависимость скорости лодки от времени")
plt.grid()
plt.show()

plt.figure()
plt.plot(t, x)
plt.xlabel("Время, с")
plt.ylabel("Путь, м")
plt.title("Зависимость пути лодки от времени")
plt.grid()
plt.show()
