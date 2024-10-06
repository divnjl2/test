import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Генерация синтетических данных
def generate_btc_like_price_data(num_points=10000, initial_price=50000, volatility=0.01, drift=0.0001, jump_prob=0.001,
                                 jump_magnitude=0.05, seed=42):
    if seed is not None:
        np.random.seed(seed)

    # Генерация цен
    prices = np.zeros(num_points)
    prices[0] = initial_price

    for i in range(1, num_points):
        random_return = np.random.normal(loc=drift, scale=volatility)
        if np.random.rand() < jump_prob:
            random_return += np.random.normal(loc=0, scale=jump_magnitude) * np.sign(np.random.randn())
        prices[i] = prices[i - 1] * np.exp(random_return)

    # Вычисляем первую и вторую производные
    first_derivative = np.gradient(prices)  # Первая производная
    second_derivative = np.gradient(first_derivative)  # Вторая производная

    # Создаём DataFrame с ценами и производными
    return pd.DataFrame({
        'Time': np.arange(num_points),
        'Price': prices,
        'First_Derivative': first_derivative,
        'Second_Derivative': second_derivative
    })

# Функция для визуализации синусоидальных данных
def visualize_price_data(price_data):
    plt.figure(figsize=(10, 6))

    # Визуализируем цену
    plt.plot(price_data['Time'], price_data['Price'], label='Price', color='blue')

    # Визуализируем первую производную
    plt.plot(price_data['Time'], price_data['First_Derivative'], label='First Derivative', color='orange')

    # Визуализируем вторую производную
    plt.plot(price_data['Time'], price_data['Second_Derivative'], label='Second Derivative', color='green')

    plt.title('Generated BTC-like Price Data')
    plt.xlabel('Time')
    plt.ylabel('Price / Derivatives')
    plt.legend()
    plt.grid(True)
    plt.show()

# Генерация данных
price_data = generate_btc_like_price_data()

# Визуализация данных
visualize_price_data(price_data)

# Функция для присвоения меток на основе уровней take-profit и stop-loss
def label_price_data(price_data, take_profit=0.02, stop_loss=0.02):
    labels = np.zeros(len(price_data))  # Массив для хранения меток

    # Логика для присвоения меток
    for i in range(len(price_data) - 1):
        current_price = price_data['Price'].iloc[i]
        tp_level = current_price * (1 + take_profit)
        sl_level = current_price * (1 - stop_loss)

        for j in range(i + 1, len(price_data)):
            future_price = price_data['Price'].iloc[j]
            if future_price >= tp_level:
                labels[i] = 1  # Take-profit сработал
                break
            elif future_price <= sl_level:
                labels[i] = -1  # Stop-loss сработал
                break

    # Добавляем столбец меток к исходному DataFrame
    price_data['Labels'] = labels

    # Возвращаем обновленный DataFrame с оригинальными данными и метками
    return price_data


# Применяем функцию для присвоения меток
labeled_data = label_price_data(price_data)

# Теперь у вас есть данные с метками и другими признаками
print(labeled_data)  # Это будет DataFrame с 5 столбцами: Time, Price, First_Derivative, Second_Derivative и Labels

# Сохраняем файл в среде
labeled_data.to_csv('labeled_data.csv', index=False)
