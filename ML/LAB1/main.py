import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import datasets
from sklearn.linear_model import LinearRegression

def get_csv_data():
    default_path = "lab1/src/student_scores.csv"
    user_path = input(f"Введите путь к CSV файлу (по умолчанию '{default_path}'): ").strip()
    path = user_path if user_path else default_path
    try:
        return pd.read_csv(path)
    except Exception as e:
        print("Ошибка при чтении файла:", e)
        return None

def select_columns(df):
    columns = list(df.columns)
    print("Найденные столбцы:", columns)
    x_col = input("Введите название столбца для X: ").strip()
    y_col = input("Введите название столбца для Y: ").strip()
    
    if len(columns) == 2:
        x_col = x_col if x_col else columns[0]
        y_col = y_col if y_col else columns[1]
    
    if x_col not in df.columns or y_col not in df.columns:
        print("Один из указанных столбцов отсутствует в файле.")
        return None, None
    
    return x_col, y_col

def show_stats(data, name):
    stats = {
        'Количество': len(data),
        'Минимум': np.min(data),
        'Максимум': np.max(data),
        'Среднее': np.mean(data)
    }
    print(f"\nСтатистика для столбца '{name}':")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

def create_scatter_plot(x, y, title, x_label, y_label):
    plt.figure(title)
    plt.scatter(x, y, color='blue', label='Данные')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

def calculate_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        print("Ошибка: деление на ноль при вычислении регрессии.")
        return None, None
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept

def plot_regression_with_line(x, y, slope, intercept, title, x_label, y_label):
    x_line = np.linspace(np.min(x), np.max(x), 100)
    y_line = slope * x_line + intercept
    plt.figure(title)
    plt.scatter(x, y, color='blue', label='Данные')
    plt.plot(x_line, y_line, color='red', label='Регрессионная прямая')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    return x_line, y_line

def plot_errors(x, y, slope, intercept, x_line, y_line, title, x_label, y_label):
    plt.figure(title)
    ax = plt.gca()
    plt.scatter(x, y, color='blue', label='Данные')
    plt.plot(x_line, y_line, color='red', label='Регрессионная прямая')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    for xi, yi in zip(x, y):
        y_pred = slope * xi + intercept
        error = abs(yi - y_pred)
        if error == 0:
            continue
        rect = patches.Rectangle(
            (xi - error/2, min(yi, y_pred)),
            error, error,
            linewidth=1, edgecolor='green',
            facecolor='green', alpha=0.3
        )
        ax.add_patch(rect)
    plt.legend()

def compare_regression_plots(X, y, feature_name, sklearn_model, custom_slope, custom_intercept):
    plt.figure(figsize=(12, 7))
    plt.scatter(X, y, color='blue', alpha=0.6, label='Фактические данные')
    
    x_line = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
    y_sklearn = sklearn_model.predict(x_line)
    y_custom = custom_slope * x_line.flatten() + custom_intercept
    
    plt.plot(x_line, y_sklearn, color='red', linewidth=2,
             label=f'Scikit-learn (m={sklearn_model.coef_[0]:.2f}, b={sklearn_model.intercept_:.2f})')
    plt.plot(x_line.flatten(), y_custom, color='green', linestyle='--', linewidth=2,
             label=f'Собственный (m={custom_slope:.2f}, b={custom_intercept:.2f})')
    
    plt.xlabel(f"Признак: {feature_name}")
    plt.ylabel("Прогрессирование диабета")
    plt.title(f"Линейная регрессия: {feature_name} vs. Диабет")
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_diabetes():
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    features = diabetes.feature_names

    print("--- Исследование набора данных Diabetes ---")
    print("\nОписание набора данных (первые 500 символов):")
    print(diabetes.DESCR[:500] + "...\n")

    print("Доступные признаки:")
    for i, name in enumerate(features):
        print(f"  {i}: {name}")

    default_idx = features.index('bmi')
    selected_idx = -1
    
    while selected_idx < 0 or selected_idx >= len(features):
        user_input = input(
            f"Введите индекс признака (0-{len(features)-1}), например {default_idx} для 'bmi' (пусто - по умолчанию): ").strip()
        if not user_input:
            selected_idx = default_idx
            print(f"Выбран признак по умолчанию: '{features[selected_idx]}'.")
            break
        try:
            selected_idx = int(user_input)
            if not 0 <= selected_idx < len(features):
                print(f"Неверный индекс. Введите число от 0 до {len(features)-1}.")
        except ValueError:
            print("Неверный ввод. Введите число.")

    feature_name = features[selected_idx]
    X_values = X[:, selected_idx]

    show_stats(X_values, feature_name)
    show_stats(y, "Прогрессирование диабета")

    sklearn_model = LinearRegression()
    sklearn_model.fit(X_values.reshape(-1, 1), y)
    sklearn_slope = sklearn_model.coef_[0]
    sklearn_intercept = sklearn_model.intercept_

    custom_slope, custom_intercept = calculate_regression(X_values, y)
    if custom_slope is None:
        return

    print("\nКоэффициенты Scikit-Learn:")
    print(f"  Наклон: {sklearn_slope:.4f}\n  Смещение: {sklearn_intercept:.4f}")
    print("\nКоэффициенты собственного алгоритма:")
    print(f"  Наклон: {custom_slope:.4f}\n  Смещение: {custom_intercept:.4f}")

    compare_regression_plots(X_values, y, feature_name, sklearn_model, custom_slope, custom_intercept)

    predictions = pd.DataFrame({
        'Фактическое': y,
        f'Sklearn ({feature_name})': sklearn_model.predict(X_values.reshape(-1, 1)),
        f'Собственный ({feature_name})': custom_slope * X_values + custom_intercept
    })
    pd.set_option('display.float_format', '{:.2f}'.format)
    print("\nПервые 20 предсказаний:")
    print(predictions.head(20).to_string())
    pd.reset_option('display.float_format')

def process_csv():
    data = get_csv_data()
    if data is None:
        return

    x_col, y_col = select_columns(data)
    if x_col is None:
        return

    x = data[x_col].values
    y = data[y_col].values

    show_stats(x, x_col)
    show_stats(y, y_col)

    create_scatter_plot(x, y, "Исходные данные", x_col, y_col)

    slope, intercept = calculate_regression(x, y)
    if slope is None:
        return

    print(f"\nПараметры прямой:\nНаклон: {slope}\nСдвиг: {intercept}")

    x_line, y_line = plot_regression_with_line(x, y, slope, intercept, "Регрессия", x_col, y_col)
    plot_errors(x, y, slope, intercept, x_line, y_line, "Ошибки", x_col, y_col)
    plt.show()

def main():
    while True:
        print("\nВыберите задачу:")
        print("1: Линейная регрессия из CSV")
        print("2: Линейная регрессия Diabetes")
        choice = input("Введите номер (1 или 2): ").strip()

        if choice == '1':
            print("\n--- Линейная регрессия на основе CSV ---")
            process_csv()
            break
        elif choice == '2':
            print("\n--- Линейная регрессия Diabetes ---")
            analyze_diabetes()
            break
        else:
            print("Неверный выбор. Введите 1 или 2.")

if __name__ == "__main__":
    main()