import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('creditcarddata.csv')

dataset.head()
dataset.shape
dataset.info()

#categorical column
dataset.select_dtypes(include='object').columns

#numerical columns
dataset.select_dtypes(include=['int64','float64']).columns

dataset.describe()

dataset.isnull().values.any()

dataset.isnull().values.sum()

dataset.isnull().any()

#filling na by mean
dataset['MINIMUM_PAYMENTS']=dataset['MINIMUM_PAYMENTS'].fillna(dataset['MINIMUM_PAYMENTS']).mean()
#dataset.dropna(inplace=True)

dataset.isnull().any()

#Remove ID variable
dataset = dataset.drop(columns='CUST_ID')

#Check correlation
corr=dataset.corr()

#plot corr
plt.figure(figsize=(10,10))
ax = sns.heatmap(corr,annot=True, cmap='coolwarm')

from copy import deepcopy
#Create train test dataset
#No Targe Variable Present

#Feature Scaling
df = dataset.copy()

from sklearn.preprocessing import  StandardScaler
sc      = StandardScaler()
dataset = sc.fit_transform(dataset)

#Finding Clustered data 
from sklearn.cluster import KMeans

wcss = []
for i in range(1,20):
  kmeans = KMeans(n_clusters=i,init='k-means++') #initialization=km++
  kmeans.fit(dataset)
  wcss.append(kmeans.inertia_)#inertia_ = sum squared distances calculated

plt.plot(range(1,20), wcss,'bx-')
plt.title('Elbow Method')
plt.xlabel('# Clusters')
plt.ylabel('WCSS') #Within clusters Sum of Squares
plt.show()

# Build a Model for finding ~8 clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8, init='k-means++', random_state=0)

# Determine Dependent Variable 
y_means = kmeans.fit_predict(dataset)

print (y_means)
y_means = y_means.reshape(len(y_means),1)
b = np.concatenate((y_means,df), axis=1)

df_final = pd.DataFrame(data=b,columns=['Cluster#','BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
       'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
       'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
       'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
       'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT',
       'TENURE'])

df_final.head()

