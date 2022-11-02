from sklearn import metrics
import numpy as np
def get_accuracy(y_true, y_pred):
    score = metrics.accuracy_score(y_true, y_pred)
    return score

def get_f1_score(y_true, y_pred, args):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true.reshape(-1)
    y_true, y_pred = y_true.tolist(), y_pred.tolist()

    score = metrics.f1_score( y_true, y_pred, 
                              average='macro')
    return score