import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
import scikitplot as skplt
import matplotlib.pyplot as plt

import sys

filename = sys.argv[1]

data = np.load(filename, allow_pickle=True).tolist()

Y_pred = []
Y_true = []
error_list =[]

for key in data.keys():
    pred = data[key]['pred_answer']
    if ('Yes' in pred) or ('yes' in pred):
        Y_pred.append('Yes')
    elif ('No' in pred) or ('no' in pred):
        Y_pred.append('No')
    else:
        error_list.append(key)
        continue
    Y_true.append(data[key]['answer'])

print(Y_pred)
print(f'len Y_pred {len(Y_pred)}')
print(Y_true)
print(f'len Y_true {len(Y_true)}')
print(error_list)

Y_pred = [1 if x=='Yes' else 0 for x in Y_pred]
Y_true = [1 if x=='Yes' else 0 for x in Y_true]

print(classification_report(Y_true, Y_pred))

acc = accuracy_score(Y_true, Y_pred)

print('\nPrecision Recall F1-Score Support Per Class : \n',precision_recall_fscore_support(Y_true, Y_pred))

fpr, tpr, thresholds = roc_curve(Y_true, Y_pred)
auc = roc_auc_score(Y_true, Y_pred)
print("False Positive Rates : {}".format(fpr))
print("True  Positive Rates : {}".format(tpr))
print("Threshols            : {}".format(thresholds))
print("AUC                  : {:.3f}".format(auc))

from sklearn.metrics import precision_recall_curve, auc,average_precision_score
precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)

p_auc = auc(recall, precision)

print("Precision : {}".format(precision))
print("Recall    : {}".format(recall))
print("Threshols : {}".format(thresholds))
print("Accuracy  : {:.3f}".format(acc))
print("AUC       : {:.3f}".format(p_auc))



