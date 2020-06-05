# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:36:51 2020

@author: sowmi
"""

import numpy as np # linear algebra mathematical and logical operations on arrays can be performed.
import pandas as pd
import os
import matplotlib.pyplot as plt 
import seaborn as sns
heart = pd.read_csv("C:\\Users\\tussh\\Documents\\Project\\heart.csv")
heart.columns
str(heart)
heart.info()
heart.describe()
heart.head(20)
heart.isnull().sum()
heart.shape

plt.hist(heart.cp)
plt.hist(heart.chol)
plt.hist(heart.trestbps)
plt.hist(heart.fbs)
plt.hist(heart.restecg)

sns.pairplot(data=heart)


fig = plt.figure(figsize = (80,25))
sns.countplot(heart.chol)

sns.boxplot(heart.chol)
heart1 = heart

age = heart['age'].value_counts().sort_values(ascending=False)
age.plot(kind="bar", figsize = (14,6), fontsize = 10,color="green")
plt.xlabel("", fontsize = 20)
plt.ylabel("", fontsize = 20)
plt.title("Count of age values", fontsize = 20)







##################################################################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt 
import seaborn as sns
df = pd.read_csv("C:\\Users\\tussh\\Documents\\Project\\heart.csv")
#Understand the data using descriptive Statistics
df.columns
df.shape
df.head(10)
df.dtypes
df.describe()
#let us see if there is any null values in our dataset.
df.isnull().sum()
#Understanding the data using Visualisation
df.hist(figsize=(12,12))
df.plot(kind='box',subplots=True,layout=(4,4),sharex=False,sharey=False,figsize=(18,18))
#Let us see that whether there is any relationship between the attributes.
df.corr()
#We cannot get a proper picture with the above analysis, let us draw a correlation graph for our better understanding.
fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(111)
cax=ax.matshow(df.corr(),vmin=-1,vmax=1)
fig.colorbar(cax)
ticks=np.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)
## Let us see that how many people are suffering from heart attack disease
df.groupby('target').size()
#people free from heart attack disease = 138
#people suffering from heart attack = 165
p_risk = (len(df.loc[(df['target']==1) ])/len(df.loc[df['target']]))*100
print("Percentage of people at risk : ", p_risk)
#Percentage of people at risk :  54.45544554455446
## Now let us see that if the gender of the person can affect him/her.
abc = pd.crosstab(df['sex'],df['target'])
abc
female_risk_percent = (len(df.loc[((df['sex']==0) & df['target']==1) ])/len(df.loc[df['sex']==0]))*100
male_risk_percent = (len(df.loc[((df['sex']==1) & df['target']==1) ])/len(df.loc[df['sex']==1]))*100
print('percentage males at risk : ',male_risk_percent)
print('percentage females at risk : ',female_risk_percent)
#percentage males at risk :  44.927536231884055
#percentage females at risk :  75.0
#We can see that the females are at greater risk of heart attack than males. Let us plot the graph between sex and target for a clearer view.
abc.plot(kind='bar', stacked=False, color=['#f5b7b1','#a9cce3'])
## We should also see that how different ages can have the risk of heart attack
#Let us draw a barplot between age and target.
xyz = pd.crosstab(df.age,df.target)
xyz.plot(kind='bar',stacked=False,figsize=(15,8))
#We can see that the people between the age of 40 to 55 are at higher risk of heart attack.¶
## Let us see that how chestpain is related with heart attack.
pqr = pd.crosstab(df.cp,df.target)
pqr
pqr.plot(kind='bar',figsize=(12,5))
#We can see that if a person has chest pain type 2 ,then he has higher chance of heart attack and if a person has chest pain type 0 , then he has a very little risk of heart attack.
## See the relationship between thal and risk of heart attack
mno = pd.crosstab(df.thal,df.target)
mno
mno.plot(kind='bar', stacked=False, color=['#2471a3','#ec7063'],figsize=(12,5))
#We can see that thal type2 can greately increase the risk of heart attack.
#We can furthermore analyze the data,but first let us do some feature selection ,create models etc for our data.
#5. Splitting data into train and test sets.
array = df.values
X = array[:, 0:13]
y = array[:, 13]

seed = 7
tsize = 0.2

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tsize, random_state=seed)

#6. Preprocess your data for Machine Learning
#From sklearn,we choose the standard scaler to preprocess our data.StandardScaler is used to transform attributes with a Gaussian Distribution with each value having mean = 0 and SD = 1
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
X_train_scale = scale.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scale)
X_test_scale =scale.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scale)

#7. Now let us create various models for training our data
#In the below code , I will be using various classification algorithms for training the model.

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

models=[]
models.append(('LR  :', LogisticRegression()))
models.append(('LDA :', LinearDiscriminantAnalysis()))
models.append(('KNN :', KNeighborsClassifier()))
models.append(('CART:', DecisionTreeClassifier()))
models.append(('NB  :', GaussianNB()))
models.append(('SVM :', SVC()))

results = []
names = []
score = 'accuracy'
seed = 7
folds = 10
X_train, X_validation, y_train, y_validation = train_test_split(X,y,test_size=0.2,random_state=seed)


for name, model in models:
    kfold = KFold(n_splits=folds,random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, scoring=score)
    results.append(cv_results)
    msg ="%s %f (%f)" % (name,cv_results.mean()*100,cv_results.std()*100)
    print(msg)
    
    
#We can see that Linear Discriminant Analysis and Logistic Regression has almost the same accuracy but the Standard deviation of LR is less than LDA so we will use LR for further eximination.
#Let us plot box graph for our different algorithms comparision.
qwerty =['LR', 'LDA', 'KNN', 'CART', 'NB', 'SVM'] 

fig = plt.figure(figsize=(10,10))
fig.suptitle("Algorithm Comparision")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(qwerty)
plt.show()

#8.At Last let us predict out test data on our trained model.
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

LR = LogisticRegression()
LR.fit(X_train, y_train)
predictions = LR.predict(X_validation)
print(accuracy_score(y_validation, predictions)*100)
print(classification_report(y_validation, predictions))
#We got an overall accuracy of 73.7% for our trained model.


