
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
import config
from utils import save_output_figure


def plot_training_results(hist_input, plot_name: str, is_frozen_layers) -> None:
  
    title = "Training Loss on {}".format(config.dataset)
    if not is_frozen_layers:
        title += " (unfrozen layers)"

    fig = plt.figure()
    n = len(hist_input.history["loss"])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, n), hist_input.history["loss"], label="train set")
    plt.plot(np.arange(0, n), hist_input.history["val_loss"], label="validation set")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Cross entropy loss")
    # plt.ylim(0, 1.5)
    plt.legend(loc="upper right")
    plt.savefig("../output/dataset-{}_model-{}_{}-Loss.png".format(config.dataset, config.model, plot_name))
    plt.show()

    title = "Training Accuracy on {}".format(config.dataset)
    if not is_frozen_layers:
        title += " (unfrozen layers)"

    fig = plt.figure()
    n = len(hist_input.history["loss"])
    plt.style.use("ggplot")
    plt.figure()

    if config.dataset == "mini-MIAS":
        plt.plot(np.arange(0, n), hist_input.history["categorical_accuracy"], label="train set")
        plt.plot(np.arange(0, n), hist_input.history["val_categorical_accuracy"], label="validation set")
    elif config.dataset == "CBIS-DDSM" or config.dataset == "mini-MIAS-binary":
        plt.plot(np.arange(0, n), hist_input.history["binary_accuracy"], label="train set")
        plt.plot(np.arange(0, n), hist_input.history["val_binary_accuracy"], label="validation set")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    plt.legend(loc="upper right")
    plt.savefig("../output/dataset-{}_model-{}_{}-Accuracy.png".format(config.dataset, config.model, plot_name))
    plt.show()





def plot_confusion_matrix(cm: np.ndarray, fmt: str, label_encoder, is_normalised: bool) -> None:
   
    title = str()
    if is_normalised:
        title = "Normalised Confusion Matrix"
        vmax = 1  # Y scale.
    elif not is_normalised:
        title = "Confusion Matrix"
        vmax = np.max(cm.sum(axis=1))  # Y scale.

    # Plot.
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, ax=ax, fmt=fmt, cmap=plt.cm.Blues, vmin=0, vmax=vmax)  # annot=True to annotate cells

    # Set labels, title, ticks and axis range.
    ax.set_xlabel('Predicted classes')
    ax.set_ylabel('True classes')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(label_encoder.classes_)
    ax.yaxis.set_ticklabels(label_encoder.classes_)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    bottom, top = ax.get_ylim()
    if is_normalised:
        save_output_figure("CM-normalised")
    elif not is_normalised:
        save_output_figure("CM")
    plt.show()
    
    



def plot_roc_curve_binary(y_true: list, y_pred: list) -> None:
    
    # Calculate fpr, tpr, and area under the curve(auc)
    # Transform y_true and y_pred from one-hot-encoding to the label-encoding.
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot.
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)  # plot roc curve
    plt.plot([0, 1], [0, 1], 'k--', color='navy', lw=2)  # plot random guess line

    # Set labels, title, ticks, legend, axis range and annotation.
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Random Guess', (.53, .48), color='navy')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    save_output_figure("ROC-binary")
    plt.show()


def plot_roc_curve_multiclass(y_true: list, y_pred: list, label_encoder) -> None:
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate fpr, tpr, area under the curve(auc) of each class.
    for i in range(label_encoder.classes_.size):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calculate macro fpr, tpr and area under the curve (AUC).
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(label_encoder.classes_))]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(label_encoder.classes_.size):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= label_encoder.classes_.size

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    # Calculate micro fpr, tpr and area under the curve (AUC).
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Plot.
    plt.figure(figsize=(8, 5))

    # Plot micro roc curve.
    plt.plot(fpr['micro'], tpr['micro'],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', lw=4)

    # Plot macro roc curve.
    plt.plot(fpr['macro'], tpr['macro'],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc['macro']),
             color='black', linestyle=':', lw=4)

    # Plot roc curve of each class.
    colors = ['#3175a1', '#e1812b', '#39923a', '#c03d3e', '#9372b2']
    for i, color in zip(range(len(label_encoder.classes_)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(label_encoder.classes_[i], roc_auc[i]))

    # Plot random guess line
    #plt.plot([0, 1], [0, 1], 'k--', color='red', lw=2)

    # Set labels, title, ticks, legend, axis range and annotation.
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Random Guess', (.53, .48), color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    save_output_figure("ROC-multi")
    plt.show()





def plot_comparison_chart(df: pd.DataFrame) -> None:
   
    title = "Accuracy Comparison"

    # Plot.
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x='paper', y='accuracy', data=df)

    # Add number at the top of the bar.
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.01, height, ha='center')

    # Set title.
    plt.title(title)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=60, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    save_output_figure(title)
    plt.show()
    
    
    


def plot_loss_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Train loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show()