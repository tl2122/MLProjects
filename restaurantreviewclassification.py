# -*- coding: utf-8 -*-
"""RestaurantReviewClassification.ipynb
"""

#Import
import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns

#dataset:https://www.kaggle.com/akram24/restaurant-reviews
dataset = pd.read_csv('/content/Restaurant_Reviews.tsv',delimiter='\t',
                      quoting=3)

dataset.head()

dataset.info()

sns.countplot(dataset['Liked'])

dataset['Length']=dataset['Review'].apply(len)

dataset

#histogram
dataset['Length'].plot(bins=100,kind='hist')

dataset.Length.describe()

#Cleaning Text
import re, nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
nsamples = len(dataset)

for i in range(nsamples):
  review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

##Create Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(max_features=1500)
corpus[0]

x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1]

features= cv.get_feature_names_out()
print(len(features),'\n',features[0:20])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from google.colab import drive
drive.mount('/content/drive')

from sklearn.naive_bayes import GaussianNB
class_nb = GaussianNB()
class_nb.fit(x_train,y_train)
y_pred = class_nb.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test,y_pred)
cm  = confusion_matrix(y_test, y_pred)
print (acc,'\n',cm)

from xgboost import XGBClassifier
class_xgb = XGBClassifier()
class_xgb.fit(x_train,y_train)
y_pred = class_xgb.predict(x_test)
acc = accuracy_score(y_test,y_pred)
cm  = confusion_matrix(y_test, y_pred)
print (acc,'\n',cm)

##Use Neural Net Classifier Sentiment Analysis
