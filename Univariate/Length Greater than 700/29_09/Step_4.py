import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

# Шаг 1: Генерация данных о цене и их разметка
def generate_btc_like_price_data(num_points=10000, initial_price=50000, volatility=0.01, drift=0.0001, jump_prob=0.001,
                                 jump_magnitude=0.05, seed=42):
    if seed is not None:
        np.random.seed(seed)

    prices = np.zeros(num_points)
    prices[0] = initial_price

    for i in range(1, num_points):
        random_return = np.random.normal(loc=drift, scale=volatility)
        if np.random.rand() < jump_prob:
            random_return += np.random.normal(loc=0, scale=jump_magnitude) * np.sign(np.random.randn())
        prices[i] = prices[i - 1] * np.exp(random_return)
    return prices

# Шаг 2: Вычисление производных
def compute_derivatives(price_data):
    first_derivative = np.gradient(price_data)
    second_derivative = np.gradient(first_derivative)
    return first_derivative, second_derivative

# Шаг 3: Разметка данных на основе тейкпрофита и стоп-лосса
def label_price_data(price_data, take_profit=0.08, stop_loss=0.04):  # Увеличение тейкпрофита и стоп-лосса
    labels = np.zeros(len(price_data))

    for i in range(len(price_data) - 1):
        current_price = price_data[i]
        tp_level = current_price * (1 + take_profit)
        sl_level = current_price * (1 - stop_loss)

        for j in range(i + 1, len(price_data)):
            future_price = price_data[j]
            if future_price >= tp_level:
                labels[i] = 1  # Покупка
                break
            elif future_price <= sl_level:
                labels[i] = -1  # Продажа
                break

    return labels

# Шаг 4: Визуализация данных (цена + метки + предсказания)
def plot_true_vs_predicted(price_data, true_labels, predicted_labels=None, title="Price with Labels"):
    fig = go.Figure()

    # Основная линия цены
    fig.add_trace(go.Scatter(x=np.arange(len(price_data)), y=price_data, mode='lines', name='Price'))

    # Истинные точки покупок и продаж
    true_buy_signals = np.where(true_labels == 1)[0]
    true_sell_signals = np.where(true_labels == -1)[0]
    fig.add_trace(go.Scatter(x=true_buy_signals, y=price_data[true_buy_signals], mode='markers',
                             marker=dict(color='green', size=6), name='True Buy'))
    fig.add_trace(go.Scatter(x=true_sell_signals, y=price_data[true_sell_signals], mode='markers',
                             marker=dict(color='red', size=6), name='True Sell'))

    # Предсказанные точки покупок и продаж (крестики), если есть предсказания
    if predicted_labels is not None:
        predicted_buy_signals = np.where(predicted_labels == 1)[0]
        predicted_sell_signals = np.where(predicted_labels == -1)[0]

        if len(predicted_buy_signals) > 0:
            fig.add_trace(go.Scatter(x=predicted_buy_signals, y=price_data[predicted_buy_signals], mode='markers',
                                     marker=dict(color='blue', size=6, symbol='x'), name='Predicted Buy'))

        if len(predicted_sell_signals) > 0:
            fig.add_trace(go.Scatter(x=predicted_sell_signals, y=price_data[predicted_sell_signals], mode='markers',
                                     marker=dict(color='orange', size=6, symbol='x'), name='Predicted Sell'))

    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price')
    fig.show()

# Шаг 5: Разделение данных на тренировочные и тестовые
def split_and_scale_data(X, y):
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return X_train_scaled, X_test_scaled, y_train, y_test

# Шаг 6: Обучение модели
def train_model(X_train_scaled, y_train):
    classifier = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
    classifier.fit(X_train_scaled, y_train)
    return classifier

# Шаг 7: Оценка модели и получение предсказаний
def evaluate_model(classifier, X_test_scaled, y_test):
    y_pred = classifier.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    return y_pred

# Основная часть программы
price_data = generate_btc_like_price_data()
first_derivative, second_derivative = compute_derivatives(price_data)
labels = label_price_data(price_data)

# Визуализация оригинальных данных с метками
plot_true_vs_predicted(price_data, labels, title="Original Price with Labels")

# Первый случай: обучаем на цене и метках
X_price_only = price_data.reshape(-1, 1)
X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(X_price_only, labels)

# Обучение модели только на цене
classifier_price = train_model(X_train_scaled, y_train)
y_pred_price = evaluate_model(classifier_price, X_test_scaled, y_test)

# Визуализация первой серии предсказаний (обучение на цене)
plot_true_vs_predicted(price_data, labels, np.concatenate([y_train, y_pred_price]), title="First Series: Price Only")

# Второй случай: обучаем на цене, первой и второй производных (3 столбца)
X_combined = np.column_stack([price_data.reshape(-1, 1), first_derivative, second_derivative])
X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(X_combined, labels)

# Обучение модели на 3 столбцах (цена + первая и вторая производные)
classifier_derivatives = train_model(X_train_scaled, y_train)
y_pred_derivatives = evaluate_model(classifier_derivatives, X_test_scaled, y_test)

# Визуализация второй серии предсказаний (обучение на цене и производных)
plot_true_vs_predicted(price_data, labels, np.concatenate([y_train, y_pred_derivatives]), title="Second Series: Price + Derivatives")
