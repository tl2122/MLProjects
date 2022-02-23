#Import
import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns

#Dataset
#https://www.kaggle.com/mlg-ulb/creditcardfraud
dataset = pd.read_csv('../KaggleData/creditcard.csv')

print("Columns\n",dataset.head(),'\n',
      "Null Values ",dataset.isnull().any().sum()
      )

from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
sc      = StandardScaler()

x = dataset.drop(columns=["Time","Class"])
y = dataset.iloc[:,-1]
#import gdb; gdb.set_trace()

x_train,  x_test, y_train, y_test = train_test_split\
                                    (x,y,test_size=0.2)
#x_train = sc.fit_transform(x_train)
#x_test  = sc.transform(x_test)

cl_rf_balanced = RandomForestClassifier(class_weight='balanced',random_state=0)
cl_rf_default  = RandomForestClassifier(random_state=0)
cl_rf_balanced.fit(x_train,y_train)
cl_rf_default .fit(x_train,y_train)

y_pred_balanced = cl_rf_balanced.predict(x_test)
y_pred_default  = cl_rf_default .predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test,y_pred_balanced)
cm  = confusion_matrix(y_test, y_pred_balanced)
print ("Balanced Classes\n","--"*10,\
        acc,'\n',cm)

acc = accuracy_score(y_test,y_pred_default)
cm  = confusion_matrix(y_test, y_pred_default)
print ("Default Dist (Unbalanced)\n","--"*10,\
        acc,'\n',cm)

