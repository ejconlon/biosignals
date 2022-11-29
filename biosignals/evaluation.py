from dataclasses import dataclass
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


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
def plot_results(name: str, results: Results):
    labels = results.y_true
    predictions_float = results.y_pred
    predictions_int = results.y_pred_int()
    fpr, tpr, _ = metrics.roc_curve(labels, predictions_float)

    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.title(f'AUC ({name})')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    cm = confusion_matrix(labels, predictions_int)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f'Confusion ({name})')
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    plt.show()

    print(cm)
    print(precision_recall_fscore_support(labels, predictions_int, average='macro'))
