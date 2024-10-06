import plotly.graph_objects as go
import pandas as pd
import numpy as np

def generate_btc_like_price_data(num_points=10000, initial_price=50000, volatility=0.01, drift=0.0001, jump_prob=0.001,
                                 jump_magnitude=0.05, seed=None):
    """
    Генерация данных о ценах с трендами, скачками и волатильностью.
    :param num_points: Количество точек данных.
    :param initial_price: Начальная цена.
    :param volatility: Волатильность, определяющая степень случайных колебаний.
    :param drift: Средний дрейф цены (направление тренда).
    :param jump_prob: Вероятность резкого скачка.
    :param jump_magnitude: Величина скачка.
    :param seed: Значение для инициализации генератора случайных чисел (если передано, генерация будет воспроизводимой).
    :return: Массив сгенерированных цен.
    """
    if seed is not None:
        np.random.seed(seed)  # Устанавливаем seed для воспроизводимости

    prices = np.zeros(num_points)
    prices[0] = initial_price

    for i in range(1, num_points):
        # Случайные колебания
        random_return = np.random.normal(loc=drift, scale=volatility)

        # Резкие скачки или падения
        if np.random.rand() < jump_prob:
            random_return += np.random.normal(loc=0, scale=jump_magnitude) * np.sign(np.random.randn())

        # Генерация цены
        prices[i] = prices[i - 1] * np.exp(random_return)

    return prices

def compute_derivatives(price_data):
    """
    Вычисление первой и второй производных массива цен.
    :param price_data: Массив данных о ценах.
    :return: Первая и вторая производные.
    """
    first_derivative = np.gradient(price_data)  # Первая производная (скорость изменения цены)
    second_derivative = np.gradient(first_derivative)  # Вторая производная (ускорение цены)
    return first_derivative, second_derivative

def label_price_data(price_data, take_profit=0.01, stop_loss=0.01):
    """
    Размечает данные цены с тремя классами (-1 - продажа, 0 - нейтральные, 1 - покупка)
    на основе тейкпрофита и стоп-лосса.
    :param price_data: Массив цен.
    :param take_profit: Процент тейкпрофита.
    :param stop_loss: Процент стоп-лосса.
    :return: Массив с лейблами.
    """
    labels = np.zeros(len(price_data))  # Инициализация массива с нулями

    for i in range(len(price_data) - 1):
        current_price = price_data[i]
        tp_level = current_price * (1 + take_profit)
        sl_level = current_price * (1 - stop_loss)

        # Проход от текущей точки до конца массива
        for j in range(i + 1, len(price_data)):
            future_price = price_data[j]

            if future_price >= tp_level:
                labels[i] = 1  # Покупка
                break
            elif future_price <= sl_level:
                labels[i] = -1  # Продажа
                break

    return labels

def save_data_to_csv(df, filename='labeled_price_data_with_derivatives.csv'):
    """
    Сохраняет датафрейм с лейблами и производными в CSV.
    :param df: Датафрейм с данными цены, лейблами и производными.
    :param filename: Имя файла для сохранения.
    """
    df.to_csv(filename, index=False)
    print(f"Данные сохранены в файл: {filename}")

def plot_labeled_data(df):
    """
    Визуализирует данные с разметкой классов интерактивно с помощью Plotly.
    :param df: Датафрейм с данными о ценах и лейблах.
    """
    fig = go.Figure()

    # Основная линия цены
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Price'], mode='lines', name='Price'))

    # Покупка (зеленые точки)
    buy_signals = df[df['Label'] == 1]
    fig.add_trace(go.Scatter(x=buy_signals['Time'], y=buy_signals['Price'],
                             mode='markers', marker=dict(color='green', size=5),
                             name='Buy'))

    # Продажа (красные точки)
    sell_signals = df[df['Label'] == -1]
    fig.add_trace(go.Scatter(x=sell_signals['Time'], y=sell_signals['Price'],
                             mode='markers', marker=dict(color='red', size=5),
                             name='Sell'))

    # Нейтральные (серые точки)
    neutral_signals = df[df['Label'] == 0]
    fig.add_trace(go.Scatter(x=neutral_signals['Time'], y=neutral_signals['Price'],
                             mode='markers', marker=dict(color='gray', size=5),
                             name='Neutral'))

    # Добавление производных в график
    fig.add_trace(go.Scatter(x=df['Time'], y=df['First_Derivative'], mode='lines', name='First Derivative', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Second_Derivative'], mode='lines', name='Second Derivative', line=dict(dash='dot')))

    fig.update_layout(title='Price Data with Labeled Buy/Sell/Neutral Signals and Derivatives',
                      xaxis_title='Time', yaxis_title='Value')

    fig.show()

# Пример использования с фиксированным seed
seed = 42  # Фиксированный seed для воспроизводимости
price_data = generate_btc_like_price_data(num_points=10000, initial_price=50000, volatility=0.01, drift=0.0001, seed=seed)

# Генерация времени
time = np.arange(len(price_data))

# Добавление лейблов
labels = label_price_data(price_data, take_profit=0.02, stop_loss=0.02)

# Вычисление производных
first_derivative, second_derivative = compute_derivatives(price_data)

# Создание датафрейма
df = pd.DataFrame({
    'Time': time,
    'Price': price_data,
    'Label': labels,
    'First_Derivative': first_derivative,
    'Second_Derivative': second_derivative
})

# Сохранение данных с производными в CSV
save_data_to_csv(df, 'labeled_price_data_with_derivatives.csv')

# Визуализация данных
plot_labeled_data(df)
