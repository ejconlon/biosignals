import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


# GRU_model = GRU(<training data>, <labels>) # to be filled in later
# LSTM_model = LSTM(<training data>, <labels>) # to be filled in later
# Bayes_model = NaiveBayes(<training data>, <labels>) # to be filled in later
# predictions = model.predict(test_data)

def evaluate_model(predictions, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    # print(fpr, tpr, thresholds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    # predict = np.argmax(actual, axis=1)
    # test_label_ = np.concatenate(labels, axis = 0)
    # print(actual)

    # test_label_ = np.argmax(actual, axis=1)

    cm = confusion_matrix(labels, predictions)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    plt.show()

    print(cm)
    print(precision_recall_fscore_support(labels, predictions, average='macro'))
