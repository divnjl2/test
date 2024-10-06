import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_custom_sine_wave(periods=5, num_points=500, amp_min=100, amp_max=200, noise_level=0,
                              amplitude_variation_factor=0.1):
    """
    513f8637-3872-4e2d-97e6-645b320785bc
    Генерирует синусоиду с изменяющейся амплитудой и кастомизируемыми параметрами.

    periods: Количество полных периодов синусоиды.
    num_points: Количество точек для синусоиды.
    amp_min: Минимальная амплитуда.
    amp_max: Максимальная амплитуда.
    noise_level: Уровень шума (чем больше, тем больше колебаний).
    amplitude_variation_factor: Коэффициент вариации амплитуды (например, 0.1 для 10% вариации).
    """
    # Параметры
    t = np.linspace(0, periods * 2 * np.pi, num_points)  # Количество полных периодов, количество точек
    amplitudes = np.random.uniform(amp_min, amp_max,
                                   size=periods)  # Случайные амплитуды в заданном диапазоне для каждого цикла

    # Генерация синусоиды с изменяющейся амплитудой
    sin_wave = np.zeros_like(t)
    points_per_period = num_points // periods  # Количество точек на каждый период
    for i in range(periods):
        start_idx = i * points_per_period
        end_idx = (i + 1) * points_per_period
        # Варьируем амплитуды в зависимости от фактора
        amplitude_variation = amplitudes[i] + np.random.normal(0, amplitudes[i] * amplitude_variation_factor)
        sin_wave[start_idx:end_idx] = amplitude_variation * np.sin(t[start_idx:end_idx])

    # Добавляем шум
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, size=t.shape)
        sin_wave += noise

    return t, sin_wave


def assign_adaptive_labels(sin_wave):
    """
    Адаптивно присваивает метки (покупка, продажа, нейтральные) в зависимости от диапазона значений синусоиды.

    Метки определяются на основе процентилей (квантилей) значений синусоиды:
    - Покупка: значения ниже 33%.
    - Нейтральная зона: значения от 34% до 66%.
    - Продажа: значения выше 67%.
    """
    low_quantile = np.percentile(sin_wave, 33)  # 33-й процентиль
    mid_quantile = np.percentile(sin_wave, 66)  # 66-й процентиль

    labels = np.zeros_like(sin_wave)

    # Покупка: значения ниже low_quantile (33%)
    labels[sin_wave <= low_quantile] = -1

    # Продажа: значения выше mid_quantile (67%)
    labels[sin_wave >= mid_quantile] = 1

    # Нейтральная зона: значения между low_quantile и mid_quantile остаются как 0 (нейтральная позиция)
    return labels, low_quantile, mid_quantile


def generate_derivatives(sin_wave, t):
    """
    Вычисляет первую и вторую производные для синусоиды.

    sin_wave: Входные данные синусоиды.
    t: Время (или индекс точек для производных).
    """
    first_derivative = np.gradient(sin_wave, t)
    second_derivative = np.gradient(first_derivative, t)
    return first_derivative, second_derivative


def save_datasets(t, sin_wave, first_derivative, second_derivative, labels):
    """
    Создает и сохраняет 4 файла с разными комбинациями данных и меток.
    """
    # Создание 4 датасетов
    df_price_label = pd.DataFrame({'Price': sin_wave, 'Label': labels})
    df_first_derivative_label = pd.DataFrame({'First_Derivative': first_derivative, 'Label': labels})
    df_second_derivative_label = pd.DataFrame({'Second_Derivative': second_derivative, 'Label': labels})
    df_first_second_derivative_label = pd.DataFrame({
        'First_Derivative': first_derivative,
        'Second_Derivative': second_derivative,
        'Label': labels
    })

    # Сохранение датасетов в файлы
    df_price_label.to_csv('Price_and_Label.csv', index=False)
    df_first_derivative_label.to_csv('First_Derivative_and_Label.csv', index=False)
    df_second_derivative_label.to_csv('Second_Derivative_and_Label.csv', index=False)
    df_first_second_derivative_label.to_csv('First_Second_Derivative_and_Label.csv', index=False)


def plot_with_labels(data, labels, derivative_name, y_label, t, low_quantile=None, mid_quantile=None, filename=None):
    """
    Визуализация данных с метками (покупка, продажа, нейтральные позиции).
    Опционально сохраняет график в файл, если указан filename.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, data, label=derivative_name, color='gray')

    # Покупка
    plt.scatter(t[labels == -1], data[labels == -1], color='blue', label='Buy (-1)')
    # Продажа
    plt.scatter(t[labels == 1], data[labels == 1], color='red', label='Sell (1)')
    # Нейтральные позиции
    plt.scatter(t[labels == 0], data[labels == 0], color='green', label='Neutral (0)')

    # Отображаем линии квантилей
    if low_quantile is not None and mid_quantile is not None:
        plt.axhline(y=low_quantile, color='blue', linestyle='--', label=f'Buy Threshold ({low_quantile:.2f})')
        plt.axhline(y=mid_quantile, color='red', linestyle='--', label=f'Sell Threshold ({mid_quantile:.2f})')

    plt.title(f'{derivative_name} with Trading Labels')
    plt.xlabel('Time')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)

    # Сохранение графика в файл, если указано имя файла
    if filename:
        plt.savefig(filename)

    plt.show()


# Основная программа

# Кастомные параметры
periods = 10  # Количество полных периодов
num_points = 1000  # Количество точек
amp_min = 50  # Минимальная амплитуда
amp_max = 800  # Максимальная амплитуда
noise_level = 20  # Уровень шума
amplitude_variation_factor = 0.80  # 20% вариация амплитуды

# Генерация синусоиды с изменяющимися параметрами и шумом
t, sin_wave = generate_custom_sine_wave(periods, num_points, amp_min, amp_max, noise_level, amplitude_variation_factor)

# Присваивание адаптивных меток
labels, low_quantile, mid_quantile = assign_adaptive_labels(sin_wave)

# Вычисление производных
first_derivative, second_derivative = generate_derivatives(sin_wave, t)

# Сохранение датасетов
save_datasets(t, sin_wave, first_derivative, second_derivative, labels)

# Визуализация с сохранением графиков
plot_with_labels(sin_wave, labels, 'Price', 'Amplitude', t, low_quantile, mid_quantile,
                 filename='Price_with_Labels.png')
plot_with_labels(first_derivative, labels, 'First Derivative', 'First Derivative Amplitude', t, low_quantile,
                 mid_quantile, filename='First_Derivative_with_Labels.png')
plot_with_labels(second_derivative, labels, 'Second Derivative', 'Second Derivative Amplitude', t, low_quantile,
                 mid_quantile, filename='Second_Derivative_with_Labels.png')

