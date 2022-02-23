# -*- coding: utf-8 -*-
"""BreastCancerDetection.ipynb
"""



"""# Part 1 : Data Preprocessing

dataset  link: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

# Import Libs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('/content/breastcancerwisconsin.csv')

dataset.head()

#Explorartion and Preprocessing
dataset.info()

dataset.select_dtypes(include='object').columns

dataset.describe()

#Missing values 
dataset.isnull().values.any()

#Categorical data 
dataset['diagnosis'].unique()

dataset = pd.get_dummies(data=dataset, drop_first=True) #One hot
dataset.head(5)

sns.countplot(x=dataset['diagnosis_M'], label='Count')
plt.show()

##Correlation
dataset_2 = dataset.drop(columns='diagnosis_M')
dataset_2.corrwith(dataset['diagnosis_M']).plot.bar(
    figsize=(20,10),title="Correlated with diagnosis_M",
    rot=45,grid=True)

corr = dataset.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr,annot=True)

#Splitting dataset
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = \
  train_test_split(x,y,test_size=0.2,random_state=0)

#Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test  = sc.transform(x_test)

"""# Part 2: Models"""

#Logistic Regression  Model 
from sklearn.linear_model import LogisticRegression
class_lr = LogisticRegression(random_state=0)
class_lr.fit(x_train,y_train)
y_pred   = class_lr.predict(x_test)

#Metrics and Cross Validation
from sklearn.metrics import accuracy_score,\
  confusion_matrix, f1_score, precision_score,recall_score
acc  = accuracy_score (y_test,y_pred)
f1   = f1_score       (y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec  = recall_score   (y_test,y_pred)
results = pd.DataFrame([['Logistic Regression',acc,f1,prec,rec]],
                        columns=['Model', 'Accuracy','F1 Score','Precision','Recall'])

cm = confusion_matrix(y_test,y_pred)
print(cm)

#Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_lr = cross_val_score(estimator=class_lr,X=x_train,\
                             y=y_train,cv=10)
print("Accuracy : {:.2f}".format(accuracies_lr.mean()*100))
print("Std      : {:.2f}".format(accuracies_lr.std()*100))

#Random Forest Model
from sklearn.ensemble import RandomForestClassifier
class_rm = RandomForestClassifier(random_state=0)
class_rm.fit(x_train,y_train)
y_pred = class_rm.predict(x_test)
acc  = accuracy_score (y_test,y_pred)
f1   = f1_score       (y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec  = recall_score   (y_test,y_pred)

results = results.append(pd.DataFrame([['Random Forest',acc,f1,prec,rec]],
                        columns=['Model', 'Accuracy','F1 Score','Precision','Recall']),
ignore_index=True)
results

accuracies_rm = cross_val_score(estimator=class_rm,X=x_train,\
                             y=y_train,cv=10)

from scipy.sparse.construct import random
#Hyperparameter Tuning
#Randomized Search to find best params
from sklearn.model_selection import RandomizedSearchCV
params = {
    'penalty':['l2','l1',],
    'C':[0.25,0.5,0.75,1,1.25,1.5,1.75,2],
    'solver':['liblinear',]
}
random_search = RandomizedSearchCV(estimator=class_lr,
                                     param_distributions=params,
                                     scoring='roc_auc',n_jobs=-1,
                                     cv=5,verbose=0)
random_search.fit(x_train,y_train)
print("Best Estimator ",random_search.best_estimator_)
print("Best SCore",random_search.best_score_)

#Apply Best Params
class_lr2 = LogisticRegression(C=1, penalty='l1', random_state=0, solver='liblinear')
class_lr2.fit(x_train,y_train)
y_pred = class_lr2.predict(x_test)

acc  = accuracy_score (y_test,y_pred)
f1   = f1_score       (y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec  = recall_score   (y_test,y_pred)
results.append(pd.DataFrame([['Logistic Regression NewParam',acc,f1,prec,rec]],
                        columns=['Model', 'Accuracy','F1 Score','Precision','Recall']),
ignore_index=True)

accuracies_lr2 = cross_val_score(estimator=class_lr2,X=x_train,\
                             y=y_train,cv=10)

for i in [accuracies_lr, accuracies_rm, accuracies_lr2]:
  print (i.mean()*100)

from sklearn.neural_network import MLPClassifier
nnclf = MLPClassifier(solver='lbfgs', activation = 'tanh',
                         alpha = 5, hidden_layer_sizes = [5,5,5],
                         random_state = 0, max_iter=1000)
nnclf.fit(x_train,y_train)
ypred = nnclf.predict(x_test)
print(accuracy_score(y_test,y_pred),'\n',
      confusion_matrix(y_test,y_pred))

from sklearn.ensemble import GradientBoostingClassifier

gbclf = GradientBoostingClassifier(random_state = 0,
                                   learning_rate=0.01,
                                   max_depth=2)
gbclf.fit(x_train, y_train)

ypred = gbclf.predict(x_test)
print(accuracy_score(y_test,y_pred),'\n',
      confusion_matrix(y_test,y_pred),'\n',
      gbclf.score(x_test,y_test))

from sklearn.naive_bayes import GaussianNB
nbclf = GaussianNB().fit(x_train, y_train)
ypred = nbclf.predict(x_test)
print(accuracy_score(y_test,y_pred),'\n',
      confusion_matrix(y_test,y_pred),'\n',
       nbclf.score(x_test,y_test)  )

