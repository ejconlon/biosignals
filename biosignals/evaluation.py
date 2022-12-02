from dataclasses import dataclass
from typing import Any, Dict, Optional
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import json


@dataclass(frozen=True)
class Confusion:
    tn: int
    fp: int
    fn: int
    tp: int

    @property
    def size(self):
        return self.tn + self.fp + self.fn + self.tp

    @property
    def accuracy(self) -> float:
        return float(self.tn + self.tp) / self.size

    @property
    def precision(self) -> float:
        return float(self.tp) / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        return float(self.tp) / (self.tp + self.fn)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2.0 * p * r / (p + r)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'size': self.size,
            'tn': self.tn,
            'fp': self.fp,
            'fn': self.fn,
            'tp': self.tp,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
        }


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

    def confusion(self) -> Confusion:
        tn, fp, fn, tp = confusion_matrix(y_true=self.y_true, y_pred=self.y_pred_int()).ravel()
        return Confusion(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


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


# Evaluate performance metrics
def eval_performance(name: str, variant: str, results: Results, dest_dir: Optional[str] = None):
    d = results.confusion().to_dict()
    for k, v in d.items():
        print(f'{name} {variant} {k}: {v}')
    with open(f'{dest_dir}/{variant}_perf.json', 'w') as f:
        json.dump(d, f, indent=2)
