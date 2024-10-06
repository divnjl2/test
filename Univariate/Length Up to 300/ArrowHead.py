# Dataset: ArrowHead, Dimensions: 1, Length:	251, Train Size: 36, Test Size: 175, Classes: 3


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.metrics import precision_score, f1_score, roc_auc_score
import time
from sklearn.preprocessing import label_binarize
from collections import Counter
from memory_profiler import memory_usage
from imblearn.over_sampling import RandomOverSampler
from itertools import cycle


# Deep Learning:
from aeon.classification.deep_learning.mlp import MLPClassifier
from aeon.classification.deep_learning.cnn import CNNClassifier
from aeon.classification.deep_learning.fcn import FCNClassifier
from sktime.classification.deep_learning.mcdcnn import MCDCNNClassifier

# Dictionary-based:
from aeon.classification.dictionary_based import (BOSSEnsemble, ContractableBOSS, IndividualBOSS,
                                                  TemporalDictionaryEnsemble, IndividualTDE, WEASEL, MUSE)

# Distance-based:
from aeon.classification.distance_based import ShapeDTW, KNeighborsTimeSeriesClassifier

# Feature-based:
from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier

# Interval-based
from aeon.classification.interval_based import (CanonicalIntervalForestClassifier, DrCIFClassifier,
                                                SupervisedTimeSeriesForest, TimeSeriesForestClassifier)

# Kernel-based:
from aeon.classification.convolution_based import RocketClassifier, Arsenal



dataset_name = "ItalyPowerDemand"  # Change this to match your dataset name

# Load the dataset
X_train_raw, y_train = load_UCR_UEA_dataset("ItalyPowerDemand", split="train", return_X_y=True)
X_test_raw, y_test = load_UCR_UEA_dataset("ItalyPowerDemand", split="test", return_X_y=True)

# Print dataset sizes and class distribution
print("Length of each time series:", X_train_raw.iloc[0, 0].size)
print("Train size:", len(y_train))
print("Test size:", len(y_test))
print("Training set class distribution:", Counter(y_train))
print("Test set class distribution:", Counter(y_test))


# Function to convert DataFrame to 2D numpy array
def dataframe_to_2darray(df):
    num_samples = df.shape[0]
    num_timesteps = len(df.iloc[0, 0])
    array_2d = np.empty((num_samples, num_timesteps))

    for i in range(num_samples):
        array_2d[i, :] = df.iloc[i, 0]

    return array_2d


# Convert and preprocess the data
scaler = TimeSeriesScalerMinMax()
X_train_processed = scaler.fit_transform(dataframe_to_2darray(X_train_raw))
X_test_processed = scaler.transform(dataframe_to_2darray(X_test_raw))  # Use the same scaler to transform test data

# Flatten each time series into a one-dimensional array for classifiers that require flat features
X_train_flat = X_train_processed.reshape((X_train_processed.shape[0], -1))
X_test_flat = X_test_processed.reshape((X_test_processed.shape[0], -1))


# Check for class imbalance
class_distribution = Counter(y_train)
min_class_size = min(class_distribution.values())
max_class_size = max(class_distribution.values())
imbalance_ratio = min_class_size / max_class_size
imbalance_threshold = 0.5

# Flag to indicate whether resampling was done
resampling_done = False

# Initialize resampled data with original data
X_train_flat_resampled, y_train_resampled = X_train_flat, y_train

# Apply oversampling if there is class imbalance
if imbalance_ratio < imbalance_threshold:
    print("Class imbalance detected. Applying RandomOverSampler...")
    ros = RandomOverSampler(random_state=0)
    X_train_flat_resampled, y_train_resampled = ros.fit_resample(X_train_flat, y_train)
    resampling_done = True


# Define a list of classifiers
classifiers = [MLPClassifier(), CNNClassifier(), FCNClassifier(), MCDCNNClassifier(),
               BOSSEnsemble(), ContractableBOSS(), IndividualBOSS(), TemporalDictionaryEnsemble(),
               IndividualTDE(), WEASEL(support_probabilities=True), MUSE(support_probabilities=True),
               ShapeDTW(), KNeighborsTimeSeriesClassifier(), Catch22Classifier(), FreshPRINCEClassifier(),
               SupervisedTimeSeriesForest(), TimeSeriesForestClassifier(),
               CanonicalIntervalForestClassifier(), DrCIFClassifier(), RocketClassifier(), Arsenal()]

# Initialize lists to store results
results = {"Classifier": [], "Execution Time": [], "Memory Usage": [], "Precision": [], "Accuracy": [],
           "F1 Score": [], "ROC-AUC Score (Macro)": [], "ROC-AUC Score (Micro)": [], "Confusion Matrix": []}


# Function to evaluate classifier
def evaluate_classifier(classifier, X_train, X_test, y_train, y_test):
    # Inner function to include both fitting and prediction for memory profiling
    def fit_and_predict():
        classifier.fit(X_train, y_train)
        return classifier.predict(X_test)

    # Measure execution time and memory usage for fitting and predicting
    start_time = time.time()
    mem_usage = memory_usage((fit_and_predict,), interval=0.1, include_children=True, retval=True)
    execution_time = time.time() - start_time
    max_mem_usage = max(mem_usage[0]) - min(mem_usage[0])  # mem_usage[0] contains the memory usage
    predicted_labels = mem_usage[1]  # mem_usage[1] contains the return value from fit_and_predict

    # Proceed with the rest of the evaluation
    precision = precision_score(y_test, predicted_labels, average='weighted')
    accuracy = accuracy_score(y_test, predicted_labels)
    f1_score_val = f1_score(y_test, predicted_labels, average='weighted')
    confusion = confusion_matrix(y_test, predicted_labels)

    # If the classifier supports probability estimates, calculate ROC AUC scores
    roc_auc_macro = roc_auc_micro = None
    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_test)
        roc_auc_macro = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        roc_auc_micro = roc_auc_score(y_test, y_prob, multi_class='ovr', average='micro')

    # Return all the metrics including memory usage
    return execution_time, max_mem_usage, precision, accuracy, f1_score_val, roc_auc_macro, roc_auc_micro, confusion


# Preparing to plot ROC-AUC curves
fpr_dict = {}
tpr_dict = {}
roc_auc_dict = {}

# Evaluate each classifier
for classifier in classifiers:
    classifier_name = type(classifier).__name__
    # Use the resampled data if resampling was done, else use the original data
    if resampling_done:
        exec_time, max_mem_usage, precision, accuracy, f1_score_val, roc_auc_macro, roc_auc_micro, confusion = \
            evaluate_classifier(classifier, X_train_flat, X_test_flat, y_train, y_test)

    else:
        exec_time, max_mem_usage, precision, accuracy, f1_score_val, roc_auc_macro, roc_auc_micro, confusion = \
            evaluate_classifier(classifier, X_train_flat_resampled, X_test_flat, y_train_resampled, y_test)


    results["Classifier"].append(classifier_name)
    results["Execution Time"].append(exec_time)
    results["Memory Usage"].append(max_mem_usage)
    results["Precision"].append(precision)
    results["Accuracy"].append(accuracy)
    results["F1 Score"].append(f1_score_val)
    results["ROC-AUC Score (Macro)"].append(roc_auc_macro)
    results["ROC-AUC Score (Micro)"].append(roc_auc_micro)
    results["Confusion Matrix"].append(confusion)

    # Print results
    print(f"{classifier_name} Execution Time: {exec_time:.2f}s")
    print(f"{classifier_name} Memory Usage: {max_mem_usage:.2f} MB")
    print(f"{classifier_name} Precision: {precision:.2f}")
    print(f"{classifier_name} Accuracy: {accuracy:.2f}")
    print(f"{classifier_name} F1 Score: {f1_score_val:.2f}")
    print(f"{classifier_name} ROC-AUC Score (Macro): {roc_auc_macro:.2f}")
    print(f"{classifier_name} ROC-AUC Score (Micro): {roc_auc_micro:.2f}")


    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_test_flat)
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


# Plot ROC-AUC Curves
# Define the number of columns and rows you want
num_cols = 3  # Fewer columns
num_rows = 7  # More rows to accommodate all classifiers, assuming 21 classifiers

# Calculate figure size dynamically based on the number of columns and rows
# Each subplot will be of size (4, 4) for example, but you can adjust this as needed
subplot_size_width = 4
subplot_size_height = 4
fig_width = subplot_size_width * num_cols
fig_height = subplot_size_height * num_rows

# Initialize the figure with the calculated dimensions
plt.figure(figsize=(fig_width, fig_height))

# Create the ROC AUC plots
for i, classifier_name in enumerate(results["Classifier"]):
    ax = plt.subplot(num_rows, num_cols, i + 1)
    for j in range(n_classes):
        ax.plot(fpr_dict[classifier_name][j], tpr_dict[classifier_name][j], lw=2,
                label=f'Class {j} (AUC = {roc_auc_dict[classifier_name][j]:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC AUC for {classifier_name}')
    ax.legend(loc="lower right")

# Adjust the spacing between subplots and the top edge of the figure
plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.9)

# Add an overall title
plt.suptitle(f'{dataset_name} ROC AUC Curves', fontsize=20, y=0.98)

# Save the figure with enough room for the suptitle
plt.tight_layout()  # This adjusts subplot params so that the subplots fit into the figure area.
plt.subplots_adjust(top=0.95)  # Adjust this value to increase the space for the title.
plt.suptitle(f"{dataset_name} ROC AUC Curves", fontsize=16)
plt.savefig(f"{dataset_name}_ROC_AUC_curves.png", bbox_inches='tight')
plt.show()

def plot_roc_auc_curves_macro(fpr_dict, tpr_dict, roc_auc_dict, classifiers, n_classes, dataset_name=dataset_name):
    plt.figure(figsize=(10, 8))

    colors = cycle(['midnightblue', 'indianred', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'mediumaquamarine', 'chocolate', 'palegreen', 'antiquewhite', 'tan', 'darkseagreen', 'aquamarine', 'cadetblue', 'powderblue', 'thistle', 'palevioletred'])

    for (classifier_name, color) in zip(classifiers, colors):
        fpr = fpr_dict[classifier_name]
        tpr = tpr_dict[classifier_name]
        roc_auc = roc_auc_dict[classifier_name]

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])  # Use np.interp instead of interp
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'macro-average ROC curve of {classifier_name} (area = {roc_auc["macro"]:.2f})',
                 color=color, linestyle='-', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name} Macro-average ROC curve per classifier')
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save the figure with the dataset name in the filename
    filename = f"{dataset_name}_macro_average_roc_curve.png"
    plt.savefig(filename)
    plt.show()
    plt.close()  # Close the figure to free memory

# Call the function with the appropriate parameters
plot_roc_auc_curves_macro(fpr_dict, tpr_dict, roc_auc_dict, results["Classifier"], n_classes)


# Function to plot results
def plot_results(results, metric, title, color):
    plt.figure(figsize=(10, 6))
    plt.bar(results["Classifier"], results[metric], color=color)
    plt.xlabel('Classifiers')
    plt.ylabel(metric)
    plt.title(title)
    plt.ylim(0, 1)
    plt.xticks(rotation=90, ha='right')
    plt.show()


def plot_results_improved(results, metric, dataset_name, color, ylabel=None):
    plt.figure(figsize=(15, 8))
    plt.bar(results["Classifier"], results[metric], color=color)
    plt.xlabel('Classifiers')
    if ylabel:
        plt.ylabel(ylabel)
    title = f"{dataset_name} {metric} Comparison"
    plt.title(title)
    if metric == "Execution Time":
        max_execution_time = max(results[metric])
        plt.ylim(0, max_execution_time * 1.1)
    else:
        plt.ylim(0, max(results[metric]) * 1.1)  # Adjust for other metrics as well

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # Save the figure
    plt.savefig(f"{dataset_name}_{metric}.png", bbox_inches='tight')
    plt.show()

# Apply the improved plotting function for each metric you want to plot
plot_results_improved(results, "Accuracy", dataset_name, "chocolate")
plot_results_improved(results, "ROC-AUC Score (Macro)", dataset_name, "saddlebrown")
plot_results_improved(results, "Execution Time", dataset_name, "sandybrown", ylabel="Time (s)")
plot_results_improved(results, "Memory Usage", dataset_name, "peachpuff", ylabel="Space (MB)")
plot_results_improved(results, "Precision", dataset_name, "peru")
plot_results_improved(results, "F1 Score", dataset_name, "sienna")


# Plot confusion matrices together with larger numbers and labels
num_classifiers = len(results["Classifier"])
num_cols = 4
num_rows = -(-num_classifiers // num_cols)  # Ceiling division

plt.figure(figsize=(20, 4 * num_rows))
for i, classifier_name in enumerate(results["Classifier"]):
    ax = plt.subplot(num_rows, num_cols, i + 1)
    cax = ax.imshow(results["Confusion Matrix"][i], interpolation='nearest', cmap=plt.cm.Oranges)
    ax.set_title(f'{classifier_name}', fontsize=14)  # Adjust title fontsize here
    cbar = plt.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=15)  # Adjust colorbar tick label size

    ax.set_xlabel('Predicted Labels', fontsize=15)  # Adjust fontsize for x-axis label
    ax.set_ylabel('True Labels', fontsize=15)  # Adjust fontsize for y-axis label

    tick_marks = np.arange(len(np.unique(y_train)))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(tick_marks, fontsize=14)  # Adjust fontsize for x-axis tick labels
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(tick_marks, fontsize=14)  # Adjust fontsize for y-axis tick labels

# Adjust the spacing of the subplots to make room for the suptitle
plt.subplots_adjust(top=0.85)  # You may need to adjust this value depending on your figure layout
plt.suptitle(f"{dataset_name} Confusion Matrices", fontsize=16)

# Save the figure with enough room for the suptitle
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # You may need to adjust these values based on your specific layout
plt.savefig(f"{dataset_name}_Confusion_Matrices.png", bbox_inches='tight')
plt.show()


def plot_roc_auc_curves_macro_with_convfs(fpr_dict, tpr_dict, roc_auc_dict, classifiers, n_classes, dataset_name="ArrowHead"):
    plt.figure(figsize=(10, 8))

    # Predefined colors, and add 'black' for the ConvFS
    colors = cycle(['midnightblue', 'indianred', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan',
                    'mediumaquamarine', 'chocolate', 'palegreen', 'antiquewhite', 'tan', 'darkseagreen', 'aquamarine',
                    'cadetblue', 'powderblue', 'thistle', 'palevioletred', 'black'])

    for (classifier_name, color) in zip(classifiers + ['ConvFS'], colors):
        if classifier_name == 'ConvFS':
            # ConvFS specific data
            fpr = {"macro": np.array([0.,         0.00819672, 0.00943396, 0.01639344, 0.01886792, 0.02459016,
 0.02830189, 0.03278689, 0.03773585, 0.04098361, 0.04918033, 0.05737705,
 0.06557377 ,0.06603774, 0.08196721, 0.09016393, 0.09433962, 0.09836066,
 0.11320755 ,0.1147541 , 0.12264151, 0.13207547, 0.14150943, 0.16393443,
 0.17213115 ,0.17924528, 0.18032787, 0.18867925, 0.19672131, 0.23584906,
 0.25409836 ,0.25471698, 0.28688525, 0.31967213, 0.3442623,  0.36885246,
 0.47540984, 0.58196721, 0.60377358, 0.62264151, 0.64150943, 0.6509434,
 0.68867925, 0.70491803, 0.72131148, 0.89622642, 0.90983607, 0.92622951,
 0.94262295, 0.99180328, 1.])}
            tpr = {"macro": np.array([0.18238994, 0.19496855, 0.22878498, 0.40488561, 0.46768754, 0.54944855,
 0.58809589, 0.5943852,  0.63303254, 0.63932185, 0.65190046, 0.66447908,
 0.6770577,  0.68671953, 0.69300884, 0.69929815, 0.7137909 , 0.72008021,
 0.76355847, 0.79500501, 0.8191596 , 0.83365236, 0.84331419, 0.8496035,
 0.86218212, 0.86701303, 0.87330234, 0.87813326, 0.88442257, 0.88925349,
 0.9018321,  0.90666302, 0.91295233, 0.91924164, 0.92553095, 0.93810956,
 0.94439887, 0.95068818, 0.95068818, 0.95068818, 0.9555191 , 0.96035001,
 0.96518093, 0.96518093, 0.96518093, 0.97001185, 0.97630116, 0.98259047,
 0.98887977, 0.99516908, 1.])}
            roc_auc = {"macro": 0.93}
        else:
            fpr = fpr_dict[classifier_name]
            tpr = tpr_dict[classifier_name]
            roc_auc = roc_auc_dict[classifier_name]

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)])) if classifier_name != 'ConvFS' else fpr['macro']
        mean_tpr = np.zeros_like(all_fpr) if classifier_name != 'ConvFS' else tpr['macro']
        if classifier_name != 'ConvFS':
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])  # Use np.interp instead of interp
            mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"]) if classifier_name != 'ConvFS' else roc_auc['macro']

        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'macro-average ROC curve of {classifier_name} (area = {roc_auc["macro"]:.2f})',
                 color=color, linestyle='-', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name} Macro-average ROC curve per classifier')
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Show the plot
    plt.show()

# Adding 'ConvFS' manually to classifiers list for demonstration. Remove or adjust in actual code if necessary.
classifiers = [classifier for classifier in results["Classifier"]]  # Assuming this list exists from your previous code
n_classes = len(np.unique(y_train))  # Assuming y_train is available from earlier
# Call the modified function
plot_roc_auc_curves_macro_with_convfs(fpr_dict, tpr_dict, roc_auc_dict, classifiers, n_classes)



import pandas as pd
# Convert the results dictionary to a DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
csv_file_name = f"{dataset_name}_results.csv"
results_df.to_csv(csv_file_name, index=False)

print(f"Results have been saved to {csv_file_name}")