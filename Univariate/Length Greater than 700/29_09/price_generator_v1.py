import numpy as np
import plotly.graph_objects as go

def generate_btc_like_price_data(num_points=10000, initial_price=50000, volatility=0.01, drift=0):
    """
    Генерация данных о ценах, визуально похожих на движение цены BTC
    :param num_points: Количество точек данных (по умолчанию 10 000)
    :param initial_price: Начальная цена (по умолчанию 50 000)
    :param volatility: Волатильность, определяющая степень колебаний
    :param drift: Средний дрейф, задающий долгосрочное направление цены (по умолчанию 0)
    :return: Массив сгенерированных цен
    """
    returns = np.random.normal(loc=drift, scale=volatility, size=num_points)
    price = initial_price * np.exp(np.cumsum(returns))
    return price

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
        tp_level = current_price * (1 + take_profit)
        sl_level = current_price * (1 - stop_loss)

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
    Визуализирует данные с разметкой классов интерактивно с помощью Plotly.
    :param price_data: Массив данных о ценах.
    :param labels: Массив лейблов.
    """
    # Основная линия цены
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(price_data)), y=price_data,
                             mode='lines', name='Price'))

    # Покупка (зеленые точки)
    buy_signals = np.where(labels == 1)[0]
    fig.add_trace(go.Scatter(x=buy_signals, y=price_data[buy_signals],
                             mode='markers', marker=dict(color='green', size=1),
                             name='Buy'))

    # Продажа (красные точки)
    sell_signals = np.where(labels == -1)[0]
    fig.add_trace(go.Scatter(x=sell_signals, y=price_data[sell_signals],
                             mode='markers', marker=dict(color='red', size=1),
                             name='Sell'))

    # Нейтральные (серые точки)
    neutral_signals = np.where(labels == 0)[0]
    fig.add_trace(go.Scatter(x=neutral_signals, y=price_data[neutral_signals],
                             mode='markers', marker=dict(color='gray', size=1),
                             name='Neutral'))

    fig.update_layout(title='Price Data with Labeled Buy/Sell/Neutral Signals',
                      xaxis_title='Time', yaxis_title='Price')

    fig.show()

# Пример использования
price_data = np.random.normal(50000, 1000, size=10000)  # Генерация случайных данных
labels = label_price_data(price_data, take_profit=0.02, stop_loss=0.02, window=50)
plot_labeled_data(price_data, labels)
