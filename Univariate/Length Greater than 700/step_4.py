import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from itertools import cycle


# Шаг 1: Загрузка вашего датасета с гибким выбором столбцов
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import label_binarize
import time
from itertools import cycle


# Шаг 1: Загрузка вашего датасета с гибким выбором столбцов
def load_custom_dataset(file_path):
    df = pd.read_csv(file_path)

    # Проверка, какие столбцы доступны
    columns = df.columns
    X = None
    y = None

    # Обработка всех возможных комбинаций признаков
    if 'First_Derivative' in columns and 'Second_Derivative' in columns and 'Label' in columns:
        print("Обрабатываем F1 + F2 и Лейбл")
        df['F1_plus_F2'] = df['First_Derivative'] + df['Second_Derivative']
        X = df[['F1_plus_F2']].values
        y = df['Label'].values
    elif 'Price' in columns and 'Label' in columns:
        print("Обрабатываем Цены и Лейбл")
        X = df[['Price']].values
        y = df['Label'].values
    elif 'First_Derivative' in columns and 'Label' in columns:
        print("Обрабатываем F(1) и Лейбл")
        X = df[['First_Derivative']].values
        y = df['Label'].values
    elif 'Second_Derivative' in columns and 'Label' in columns:
        print("Обрабатываем F(2) и Лейбл")
        X = df[['Second_Derivative']].values
        y = df['Label'].values
    else:
        raise ValueError("Набор данных не содержит нужных столбцов для обработки")

    return X, y


# Путь к вашему датасету
file_path = 'Price_and_Label.csv'  # Укажите путь к вашему файлу
X_raw, y = load_custom_dataset(file_path)

# Шаг 2: Разделение на обучающую и тестовую выборки
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.3, random_state=42, stratify=y)

# Преобразование и предобработка данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Функция для оценки классификатора
def evaluate_classifier(classifier, X_train, X_test, y_train, y_test):
    start_time = time.time()
    classifier.fit(X_train, y_train)
    predicted_labels = classifier.predict(X_test)
    execution_time = time.time() - start_time

    precision = precision_score(y_test, predicted_labels, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test, predicted_labels)
    f1_score_val = f1_score(y_test, predicted_labels, average='weighted', zero_division=0)
    confusion = confusion_matrix(y_test, predicted_labels)

    roc_auc_macro = roc_auc_micro = None
    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
        roc_auc_macro = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
        roc_auc_micro = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='micro')

    return execution_time, precision, accuracy, f1_score_val, roc_auc_macro, roc_auc_micro, confusion, y_prob


# Функция визуализации вероятностей
def plot_with_predictions(data, labels, y_prob, t, low_quantile=None, mid_quantile=None, filename=None):
    plt.figure(figsize=(12, 8))

    # Верхний график: цена и метки
    plt.subplot(2, 1, 1)
    plt.plot(t, data, label='Price', color='gray')

    # Покупка
    plt.scatter(t[labels == -1], data[labels == -1], color='blue', label='Buy (-1)', s=20, alpha=0.7)
    # Продажа
    plt.scatter(t[labels == 1], data[labels == 1], color='red', label='Sell (1)', s=20, alpha=0.7)
    # Нейтральные позиции
    plt.scatter(t[labels == 0], data[labels == 0], color='green', label='Neutral (0)', s=20, alpha=0.7)

    # Линии для порогов
    if low_quantile is not None and mid_quantile is not None:
        plt.axhline(y=low_quantile, color='blue', linestyle='--', label=f'Buy Threshold ({low_quantile:.2f})')
        plt.axhline(y=mid_quantile, color='red', linestyle='--', label=f'Sell Threshold ({mid_quantile:.2f})')

    plt.title(f'Price with Trading Labels')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Нижний график: вероятности предсказаний
    plt.subplot(2, 1, 2)
    plt.plot(t, y_prob[:, 0], label='Probability of Buy (-1)', color='blue', linestyle='--')
    plt.plot(t, y_prob[:, 1], label='Probability of Neutral (0)', color='green', linestyle='--')
    plt.plot(t, y_prob[:, 2], label='Probability of Sell (1)', color='red', linestyle='--')

    plt.title('Prediction Probabilities')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)

    # Сохранение графика в файл
    if filename:
        plt.savefig(filename)

    plt.show()


# Инициализация классификаторов
classifiers = [
    MLPClassifier(max_iter=1000),
    # Добавьте другие классификаторы по необходимости
]

# Оценка и визуализация для каждого классификатора
for classifier in classifiers:
    classifier_name = type(classifier).__name__

    exec_time, precision, accuracy, f1_score_val, roc_auc_macro, roc_auc_micro, confusion, y_prob = \
        evaluate_classifier(classifier, X_train_scaled, X_test_scaled, y_train, y_test)

    # Вывод результатов
    print(f"{classifier_name} Execution Time: {exec_time:.2f}s")
    print(f"{classifier_name} Precision: {precision:.2f}")
    print(f"{classifier_name} Accuracy: {accuracy:.2f}")
    print(f"{classifier_name} F1 Score: {f1_score_val:.2f}")
    print(f"{classifier_name} ROC-AUC Score (Macro): {roc_auc_macro}")
    print(f"{classifier_name} ROC-AUC Score (Micro): {roc_auc_micro}")

    # Визуализация цены и вероятностей предсказаний
    plot_with_predictions(X_test_raw.flatten(), y_test, y_prob, np.arange(len(y_test)),
                          filename=f"{classifier_name}_price_with_predictions.png")

