import numpy as np
import matplotlib.pyplot as plt

xB, yB = 6, 4  

def f(t):
    return (1 - np.cos(t)) - (yB / xB) * (t - np.sin(t))

a, b = 1e-6, 2 * np.pi

for _ in range(100):
    c = (a + b) / 2
    if f(a) * f(c) < 0:
        b = c
    else:
        a = c

tB = (a + b) / 2
R = xB / (tB - np.sin(tB))

# Параметрические уравнения циклоиды
t = np.linspace(0, tB, 200)
x = R * (t - np.sin(t))
y = -R * (1 - np.cos(t))  

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='Брахистохрона (циклоида)')
plt.plot([0, xB], [0, -yB], 'ro', label='Точки A и B')
plt.plot([0, xB], [0, -yB], 'r--', alpha=0.5)
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.title('Решение задачи о брахистохроне')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print(f"R = {R:.4f}, tB = {tB:.4f} рад = {np.degrees(tB):.1f}°")
print(f"Проверка: x(tB) = {R*(tB - np.sin(tB)):.6f}, y(tB) = {-R*(1 - np.cos(tB)):.6f}")