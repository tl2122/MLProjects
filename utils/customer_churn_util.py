
#cross-validation score metric
def cv_scoring(classifier,x,y):
    return accuracy_score(y,classifier.predict(x))

from sklearn.metrics import roc_curve, recall_score,roc_auc_score,auc
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_predict

import numpy as np
def clf_performance(yt_,ypred_,ypred_pr):
    y_train = yt_
    y_pred  = ypred_


    #print (y_train,y_pred,ypred_pr)
    #print (len(y_pred))
    cm          = confusion_matrix(y_train, y_pred)
    fnr         = cm[0][0]/np.sum(cm[0])
    fpr, tpr, _ = roc_curve(y_train, ypred_pr)

    sc = [
        accuracy_score(y_train, y_pred),
        recall_score(y_train, y_pred),
        roc_auc_score(y_train, y_pred), fpr, tpr,cm,fnr
    ]
    return sc

def get_scores(clf_,xtr_,ytr_,xts_,yts_):
  clf_.fit(xtr_,ytr_)
  y_pred_tr = clf_.predict(xtr_)
  y_pred    = clf_.predict(xts_)
  ##-- Alternate Prediction
  y_pred_pr = cross_val_predict(estimator=clf_,
                                X=xts_,
                                y=yts_,cv=5,
                                method='predict_proba')[:, 1]
  scores    = clf_performance(yts_,y_pred,y_pred_pr)
  scores_tr = clf_performance(ytr_,y_pred_tr,y_pred_tr)
  return (scores,scores_tr)
