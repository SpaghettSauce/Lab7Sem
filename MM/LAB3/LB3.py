import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# начальные данные
x0 = 100          # начальный размер популяции
t_start = 0       # начало наблюдения
t_end = 20        # конец наблюдения (часы)
t_eval = np.linspace(t_start, t_end, 300)

# дифференциальное уравнение
def population_model(t, x):
    return x / 5

# численное решение дифура
solution = solve_ivp(
    population_model,
    [t_start, t_end],
    [x0],
    t_eval=t_eval
)

# аналитическое решение
x_analytical = x0 * np.exp(t_eval / 5)

# построение графика
plt.figure(figsize=(8, 5))
plt.plot(t_eval, solution.y[0], label='Численное решение', linewidth=2)
plt.plot(t_eval, x_analytical, '--', label='Аналитическое решение', linewidth=2)

plt.xlabel('Время t (часы)')
plt.ylabel('Размер популяции x(t)')
plt.title('Рост популяции бактерий')
plt.grid(True)
plt.legend()
plt.show()
