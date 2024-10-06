import numpy as np
import matplotlib.pyplot as plt


def generate_btc_like_price_data(num_points=10000, initial_price=50000, volatility=0.01, drift=0):
    """
    Генерация данных о ценах, визуально похожих на движение цены BTC
    :param num_points: Количество точек данных (по умолчанию 10 000)
    :param initial_price: Начальная цена (по умолчанию 50 000)
    :param volatility: Волатильность, определяющая степень колебаний
    :param drift: Средний дрейф, задающий долгосрочное направление цены (по умолчанию 0)
    :return: Массив сгенерированных цен
    """
    # Генерация случайных процентных изменений цен с нормальным распределением
    returns = np.random.normal(loc=drift, scale=volatility, size=num_points)

    # Накопление логарифмических возвратов для генерации цен
    price = initial_price * np.exp(np.cumsum(returns))

    return price


def plot_price_data(price_data, interval=1):
    """
    Визуализация сгенерированных данных
    :param price_data: Массив данных о ценах
    :param interval: Интервал дискретизации (в секундах)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, len(price_data) * interval, interval), price_data)
    plt.title('Смоделированные данные цены, похожие на BTC')
    plt.xlabel('Время (в секундах)')
    plt.ylabel('Цена')
    plt.show()


def label_price_data(price_data, take_profit=0.01, stop_loss=0.01, window=50):
    """
    Размечает данные цены с тремя классами (-1 - продажа, 0 - нейтральные, 1 - покупка)
    на основе тейкпрофита и стоп-лосса.

    :param price_data: Массив цен.
    :param take_profit: Процент тейкпрофита (по умолчанию 1%).
    :param stop_loss: Процент стоп-лосса (по умолчанию 1%).
    :param window: Размер окна для анализа цен.
    :return: Массив с лейблами.
    """
    labels = np.zeros(len(price_data))  # Инициализация массива с нулями

    for i in range(len(price_data) - window):
        current_price = price_data[i]

        # Устанавливаем уровни тейкпрофита и стоп-лосса
        tp_level = current_price * (1 + take_profit)
        sl_level = current_price * (1 - stop_loss)

        # Проверяем следующие значения цены в пределах окна
        for j in range(1, window):
            future_price = price_data[i + j]

            if future_price >= tp_level:
                labels[i] = 1  # Покупка
                break
            elif future_price <= sl_level:
                labels[i] = -1  # Продажа
                break

    return labels


def plot_labeled_data(price_data, labels):
    """
    Визуализирует данные с разметкой классов.

    :param price_data: Массив данных о ценах.
    :param labels: Массив лейблов.
    """
    plt.figure(figsize=(12, 6))

    # Визуализация цены
    plt.plot(price_data, label='Price', color='blue')

    # Визуализация лейблов
    buy_signals = np.where(labels == 1)[0]
    sell_signals = np.where(labels == -1)[0]
    neutral_signals = np.where(labels == 0)[0]

    # Покупка (зелёные точки)
    plt.scatter(buy_signals, price_data[buy_signals], color='green', label='Buy', marker='^', s=100)
    # Продажа (красные точки)
    plt.scatter(sell_signals, price_data[sell_signals], color='red', label='Sell', marker='v', s=100)
    # Нейтральные (серые точки)
    plt.scatter(neutral_signals, price_data[neutral_signals], color='gray', label='Neutral', marker='o', s=50)

    plt.title('Price Data with Labeled Buy/Sell/Neutral Signals')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# Пример использования
price_data = np.random.normal(50000, 1000, size=10000)  # Генерация случайных данных

# Разметка данных
labels = label_price_data(price_data, take_profit=0.02, stop_loss=0.02, window=50)

# Визуализация данных
plot_labeled_data(price_data, labels)
