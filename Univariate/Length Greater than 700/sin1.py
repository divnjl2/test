import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import Counter
from sklearn.metrics import (
    precision_score, accuracy_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from itertools import cycle

# Импорт классификаторов из aeon
from aeon.classification.deep_learning import CNNClassifier, FCNClassifier
from sktime.classification.deep_learning.mcdcnn import MCDCNNClassifier

from aeon.classification.dictionary_based import (
    BOSSEnsemble, ContractableBOSS, IndividualBOSS, TemporalDictionaryEnsemble, IndividualTDE
)
from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier
from aeon.classification.interval_based import (
    CanonicalIntervalForestClassifier, DrCIFClassifier, SupervisedTimeSeriesForest, TimeSeriesForestClassifier
)
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier, ShapeDTW
from sktime.classification.kernel_based import Arsenal, RocketClassifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.transformations.series.summarize import SummaryTransformer

# Укажите название вашего датасета
dataset_name = "price_label.csv"  # Замените на ваше название

# Шаг 1: Загрузка вашего датасета с гибким выбором столбцов
def load_custom_dataset(file_path):
    df = pd.read_csv(file_path)

    # Проверка, какие столбцы доступны
    columns = df.columns
    X = None
    y = None

    # 4) F1 + F2 и лейбл: Сначала проверяем это условие, чтобы оно имело приоритет
    if 'First_Derivative' in columns and 'Second_Derivative' in columns and 'Label' in columns:
        print("Обрабатываем F1 + F2 и Лейбл")
        df['F1_plus_F2'] = df['First_Derivative'] + df['Second_Derivative']  # Создаем новый столбец
        X = df[['F1_plus_F2']].values  # Используем сумму первых и вторых производных
        y = df['Label'].values    # Метки классов

    # 1) Цены и лейбл
    elif 'Price' in columns and 'Label' in columns:
        print("Обрабатываем Цены и Лейбл")
        X = df[['Price']].values  # Извлекаем только цену
        y = df['Label'].values    # Метки классов

    # 2) F(1) и лейбл
    elif 'First_Derivative' in columns and 'Label' in columns:
        print("Обрабатываем F(1) и Лейбл")
        X = df[['First_Derivative']].values  # Извлекаем только первый производный
        y = df['Label'].values    # Метки классов

    # 3) F(2) и лейбл
    elif 'Second_Derivative' in columns and 'Label' in columns:
        print("Обрабатываем F(2) и Лейбл")
        X = df[['Second_Derivative']].values  # Извлекаем только второй производный
        y = df['Label'].values    # Метки классов

    # Если ни один из условий не сработал
    else:
        raise ValueError("Набор данных не содержит нужных столбцов для обработки")

    return X, y


# Путь к вашему датасету (замените на путь к вашему файлу)
file_path = 'labeled_price_data.csv'  # Замените на путь к вашему файлу
X_raw, y = load_custom_dataset(file_path)

# Шаг 2: Разделение на обучающую и тестовую выборки
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.8, random_state=42, stratify=y)

# Вывод информации о размерах выборок и распределении классов
print("Length of each time series:", X_train_raw.shape[1])
print("Train size:", len(y_train))
print("Test size:", len(y_test))
print("Training set class distribution:", Counter(y_train))
print("Test set class distribution:", Counter(y_test))

# Функция для преобразования массива numpy в DataFrame с вложенными сериями
def array_to_nested_dataframe(X):
    num_samples, num_features = X.shape
    nested_df = pd.DataFrame()
    for i in range(num_features):
        nested_df[f'dim_{i}'] = [pd.Series(X[j, i]) for j in range(num_samples)]
    return nested_df

# Преобразование и предобработка данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Преобразование данных в формат DataFrame с вложенными сериями
X_train_processed = array_to_nested_dataframe(X_train_scaled)
X_test_processed = array_to_nested_dataframe(X_test_scaled)

# Преобразование данных в 2D numpy массив для классификаторов, требующих плоских признаков
X_train_flat = X_train_scaled.reshape((X_train_scaled.shape[0], -1))
X_test_flat = X_test_scaled.reshape((X_test_scaled.shape[0], -1))

# Проверка на несбалансированность классов
class_distribution = Counter(y_train)
min_class_size = min(class_distribution.values())
max_class_size = max(class_distribution.values())
imbalance_ratio = min_class_size / max_class_size
imbalance_threshold = 0.5

# Флаг для указания, был ли применен ресемплинг
resampling_done = False

# Инициализация данных для обучения с исходными данными
X_train_flat_resampled, y_train_resampled = X_train_flat, y_train
X_train_processed_resampled = X_train_processed



# Определение списка классификаторов
classifiers = [
    MLPClassifier(),
]

# Инициализация словарей для хранения результатов
results = {"Classifier": [], "Execution Time": [], "Precision": [], "Accuracy": [], "F1 Score": [], "ROC-AUC Score (Macro)": [], "ROC-AUC Score (Micro)": [], "Confusion Matrix": []}

# Функция для оценки классификатора
def evaluate_classifier(classifier, X_train, X_test, y_train, y_test):
    def fit_and_predict():
        classifier.fit(X_train, y_train)
        return classifier.predict(X_test)

    start_time = time.time()
    predicted_labels = fit_and_predict()
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

    return execution_time, None, precision, accuracy, f1_score_val, roc_auc_macro, roc_auc_micro, confusion

# Подготовка к построению ROC-AUC кривых
fpr_dict = {}
tpr_dict = {}
roc_auc_dict = {}

# Оценка каждого классификатора
for classifier in classifiers:
    classifier_name = type(classifier).__name__

    # Определяем, какие данные использовать в зависимости от классификатора
    if classifier_name in ['MLPClassifier']:
        X_train_current = X_train_flat_resampled if resampling_done else X_train_flat
        X_test_current = X_test_flat
    else:
        X_train_current = X_train_processed_resampled if resampling_done else X_train_processed
        X_test_current = X_test_processed

    # Оценка текущего классификатора
    try:
        exec_time, max_mem_usage, precision, accuracy, f1_score_val, roc_auc_macro, roc_auc_micro, confusion = \
            evaluate_classifier(classifier, X_train_current, X_test_current, y_train_resampled, y_test)


        # Сохранение результатов в словарь
        results["Classifier"].append(classifier_name)
        results["Execution Time"].append(exec_time)
        results["Precision"].append(precision)
        results["Accuracy"].append(accuracy)
        results["F1 Score"].append(f1_score_val)
        results["ROC-AUC Score (Macro)"].append(roc_auc_macro)
        results["ROC-AUC Score (Micro)"].append(roc_auc_micro)
        results["Confusion Matrix"].append(confusion)

        # Вывод результатов
        print(f"{classifier_name} Execution Time: {exec_time:.2f}s")
        print(f"{classifier_name} Precision: {precision:.2f}")
        print(f"{classifier_name} Accuracy: {accuracy:.2f}")
        print(f"{classifier_name} F1 Score: {f1_score_val:.2f}")
        print(f"{classifier_name} ROC-AUC Score (Macro): {roc_auc_macro}")
        print(f"{classifier_name} ROC-AUC Score (Micro): {roc_auc_micro}")

        if hasattr(classifier, "predict_proba"):
            y_prob = classifier.predict_proba(X_test_current)
            y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
            n_classes = y_test_bin.shape[1]

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            fpr_dict[classifier_name] = fpr
            tpr_dict[classifier_name] = tpr
            roc_auc_dict[classifier_name] = roc_auc
    except Exception as e:
        print(f"Error evaluating classifier {classifier_name}: {e}")

# Построение ROC-AUC кривых
def plot_roc_auc_curves_macro(fpr_dict, tpr_dict, roc_auc_dict, classifiers, n_classes, dataset_name=dataset_name):
    plt.figure(figsize=(10, 8))

    colors = cycle(['midnightblue', 'indianred', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan'])

    for (classifier_name, color) in zip(classifiers, colors):
        if classifier_name in fpr_dict:
            fpr = fpr_dict[classifier_name]
            tpr = tpr_dict[classifier_name]
            roc_auc = roc_auc_dict[classifier_name]

            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            plt.plot(fpr["macro"], tpr["macro"],
                     label=f'{classifier_name} (area = {roc_auc["macro"]:.2f})',
                     color=color, linestyle='-', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name} Macro-average ROC curve per classifier')
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Сохранение графика
    filename = f"{dataset_name}_macro_average_roc_curve.png"
    plt.savefig(filename)
    plt.show()
    plt.close()

# Вызываем функцию построения ROC-AUC кривых
n_classes = len(np.unique(y_train))
plot_roc_auc_curves_macro(fpr_dict, tpr_dict, roc_auc_dict, results["Classifier"], n_classes)

# Функция для построения результатов
def plot_results_improved(results, metric, dataset_name, color, ylabel=None):
    plt.figure(figsize=(15, 8))
    plt.bar(results["Classifier"], results[metric], color=color)
    plt.xlabel('Classifiers')
    if ylabel:
        plt.ylabel(ylabel)
    title = f"{dataset_name} {metric} Comparison"
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_{metric}.png", bbox_inches='tight')
    plt.show()

# Построение графиков для метрик
plot_results_improved(results, "Accuracy", dataset_name, "chocolate", ylabel="Accuracy")
plot_results_improved(results, "ROC-AUC Score (Macro)", dataset_name, "saddlebrown", ylabel="ROC-AUC Score (Macro)")
plot_results_improved(results, "Execution Time", dataset_name, "sandybrown", ylabel="Time (s)")
plot_results_improved(results, "Precision", dataset_name, "peru", ylabel="Precision")
plot_results_improved(results, "F1 Score", dataset_name, "sienna", ylabel="F1 Score")

# Построение матриц ошибок
num_classifiers = len(results["Classifier"])
num_cols = 4  # Можно настроить
num_rows = -(-num_classifiers // num_cols)  # Округление вверх

plt.figure(figsize=(20, 5 * num_rows))
for i, classifier_name in enumerate(results["Classifier"]):
    if results["Confusion Matrix"][i] is not None:
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(results["Confusion Matrix"][i], interpolation='nearest', cmap=plt.cm.Oranges)
        plt.title(f'{classifier_name}')
        plt.colorbar()
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        tick_marks = np.arange(n_classes)
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(f"{dataset_name} Confusion Matrices", fontsize=16)
plt.savefig(f"{dataset_name}_Confusion_Matrices.png", bbox_inches='tight')
plt.show()


def plot_with_predictions(data, labels, y_prob, t, low_quantile=None, mid_quantile=None, filename=None):
    """
    Визуализация цены, меток и вероятностей предсказания.
    """
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


# Пример использования внутри цикла классификаторов
for classifier in classifiers:
    classifier_name = type(classifier).__name__

    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_test_current)

        # Визуализация цены и вероятностей предсказаний
        plot_with_predictions(X_test_current.flatten(), y_test, y_prob, np.arange(len(y_test)),
                              filename=f"{classifier_name}_price_with_predictions.png")

