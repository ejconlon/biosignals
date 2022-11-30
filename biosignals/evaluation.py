from dataclasses import dataclass
from typing import Optional
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import json


# Results (actual classes vs predicted classes)
@dataclass(frozen=True)
class Results:
    y_true: np.ndarray
    y_pred: np.ndarray

    # Round and cast y_pred to int
    def y_pred_int(self) -> np.ndarray:
        if np.issubdtype(self.y_pred.dtype, np.integer):
            return self.y_pred
        else:
            return np.rint(self.y_pred).astype(int)

    def accuracy(self) -> float:
        size = self.y_true.shape[0]
        tn, _, _, tp = confusion_matrix(y_true=self.y_true, y_pred=self.y_pred_int()).ravel()
        return float(tn + tp) / size


# Plot classification results (AUC and confusion)
# If dest_dir is None then will show plots,
# Otherwise will just write images into that directory
def plot_results(name: str, variant: str, results: Results, dest_dir: Optional[str] = None):
    labels = results.y_true
    predictions_float = results.y_pred
    predictions_int = results.y_pred_int()
    fpr, tpr, _ = metrics.roc_curve(labels, predictions_float)

    roc_auc = metrics.auc(fpr, tpr)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'AUC ({name} {variant})')
    ax.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    if dest_dir is None:
        fig.show()
    else:
        fig.savefig(f'{dest_dir}/{variant}_auc.png')

    cm = confusion_matrix(labels, predictions_int)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'Confusion ({name} {variant})')
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    if dest_dir is None:
        fig.show()
    else:
        fig.savefig(f'{dest_dir}/{variant}_conf.png')

    print(cm)
    print(precision_recall_fscore_support(labels, predictions_int, average='macro'))


# Evaluate performance metrics
def eval_performance(name: str, variant: str, results: Results, dest_dir: Optional[str] = None):
    acc = results.accuracy()
    print(f'{name} {variant} accuracy', acc)
    with open(f'{dest_dir}/{variant}_perf.json', 'w') as f:
        json.dump({'accuracy': acc}, f, indent=2)
