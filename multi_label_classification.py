import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix


def plot_binary_confusion_matrices(y_true, y_pred, class_names, fig, axes, label_names=["N", "Y"], 
                                   cmap='Blues', fontsize=12, cbar=False):
    """
    Plot binary confusion matrices sequentially by each classes.
    
    ----------
    Parameters
    y_true : {array-like, sparse matrix} of shape (n_samples, n_outputs) or \
             (n_samples,) Ground truth (correct) target values.
    y_pred : {array-like, sparse matrix} of shape (n_samples, n_outputs) or \
             (n_samples,) Estimated targets as returned by a classifier.
    class_names : array-like of shape (n_outputs,).
    fig, axes : fig, axes = plt.subplots(...), axes.flatten().shape == n_outputs.
    label_names : label names of binary confusion matrices, default=["Y", "N"].
    cmap : matplotlib colormap name or object, or list of colors, default='Blues'.
    fontsize : frontsize, default=12.
    cbar : Whether to draw a colorbar, default=False.
    
    -------
    Returns
    cms : ndarray of shape (n_outputs, 2, 2)
    """
    if len(axes.flatten()) < len(class_names):
        raise ValueError(f"There is {len(class_names)} classes, but only {len(axes.flatten())} displayed.")
    cms = multilabel_confusion_matrix(y_true, y_pred)
    sns.set(font_scale=fontsize/10)
    for cn, cm, ax, in zip(class_names, cms, axes.flatten()):
        df_cm = pd.DataFrame(cm, index=label_names, columns=label_names)
        heatmap = sns.heatmap(df_cm, cmap=cmap, annot=True, fmt="d", cbar=cbar, ax=ax)
        ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=fontsize)
        ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        ax.set_xlabel('Predicted label', fontsize=1.2*fontsize)
        ax.set_ylabel('True label', fontsize=1.2*fontsize)
        ax.set_title(f'Confusion Matrix of Class - {cn}', fontsize=1.5*fontsize)
    fig.tight_layout()
    return cms
    
    
def plot_multi2single_confusion_matrx(y_true, y_pred, class_names, label_names=None, normalize=None, 
                                      cmap='Blues', fontsize=12, cbar=False):
    """
    Plot confusion matrices of multilabel classification as normal classification.
    
    ----------
    Parameters
    y_true : {array-like, sparse matrix} of shape (n_samples, n_outputs) or \
            (n_samples,) Ground truth (correct) target values.
    y_pred : {array-like, sparse matrix} of shape (n_samples, n_outputs) or \
            (n_samples,) Estimated targets as returned by a classifier.
    label_names : array-like of shape (n_multilabel_classes), default=None.
    class_names : array-like of shape (n_singlelabel_classes,), default=None.
    normalize : {‘true’, ‘pred’, ‘all’}, default=None, Normalizes confusion matrix over the true (rows), \
                predicted (columns) conditions or all the population. \
                If None, confusion matrix will not be normalized.
    cmap : matplotlib colormap name or object, or list of colors, default='Blues'.
    fontsize : frontsize, default=12.
    cbar : Whether to draw a colorbar, default=False.
    
    -------
    Returns
    cm : ndarray of shape (n_multilabel_classes, n_multilabel_classes)
    """
    label_y_true, label_y_pred = [], []
    for yt in y_true:
        label_y_true.append(str(yt))
    for yp in y_pred:
        label_y_pred.append(str(yp))
    if not label_names:
        label_names = sorted(set(label_y_true+label_y_pred))
    cm = confusion_matrix(label_y_true, label_y_pred, labels=label_names, normalize=normalize)
    df_cm = pd.DataFrame(cm, index=label_names, columns=label_names)
    sns.set(font_scale=fontsize/10)
    heatmap = sns.heatmap(df_cm, cmap=cmap, annot=True, fmt="d", cbar=cbar)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='center', fontsize=fontsize)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.set_xlabel('Predicted label', fontsize=1.2*fontsize)
    heatmap.set_ylabel('True label', fontsize=1.2*fontsize)
    heatmap.set_title(f'Confusion Matrix as Single Label type, class_names: {class_names}', fontsize=1.5*fontsize)
    return cm