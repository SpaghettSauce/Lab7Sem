import math

g = 9.8
m = 0.6

# Ввод высоты H > 0
while True:
    try:
        H = float(input("Введите высоту столба (в метрах):\n"))
        if H > 0:
            break
    except ValueError:
        pass
    print("Высота должна быть положительным числом.")

# Ввод сторон прямоугольника для вычисления площади P
while True:
    try:
        tmp = float(input("Введите длину прямоугольного сечения (в метрах):\n"))
        tmp1 = float(input("Введите ширину прямоугольного сечения (в метрах):\n"))
        if tmp > 0 and tmp1 > 0:
            break
    except ValueError:
        pass
    print("Длина и ширина должны быть положительными числами.")

P = tmp * tmp1
print(f"Площадь поперечного сечения: {P}")

# Ввод скорости V > 0 (в км/ч), конвертируем в м/с
while True:
    try:
        V = float(input("Введите скорость движения воды в трубе (в км/ч):\n"))
        if V > 0:
            break
    except ValueError:
        pass
    print("Скорость должна быть положительным числом.")

V = V * 1000 / 3600  # Переводим из км/ч в м/с

# Ввод площади S (в см²), конвертируем в м² и проверяем условие S > P
while True:
    try:
        S_cm2 = float(input("Введите площадь отверстия в конце трубы (в см²):\n"))
        if S_cm2 > 0:
            S = S_cm2 / 10000  # Переводим см² в м²
            if S > P:
                break
    except ValueError:
        pass
    print("Площадь должна быть положительной и больше площади сечения трубы (в м²).")

# Вычисление времени
# Проверка аргумента логарифма на положительность
arg_log = V - m * S * math.sqrt(2 * g * H)
if arg_log <= 0:
    print("Ошибка: аргумент логарифма должен быть положительным. Проверьте введённые данные.")
else:
    term1 = (P * V * math.log(V)) / (g * m * m * S * S)
    term2 = (2 * P) / (math.sqrt(2 * g) * m * S)
    term3 = math.sqrt(H) + (V / (math.sqrt(2 * g) * m * S)) * math.log(arg_log)
    time = term1 - term2 * term3

    print("Время опустошения бака:")
    print(f"{time} сек")

# Ожидание ввода перед завершением (аналогично cin >> end;)
input("Нажмите Enter для выхода...")