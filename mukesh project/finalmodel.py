# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('gender_voice_dataset.csv')
df.describe()

df.head()

df.isnull().values.any()

#extract features and labels
df=df.sample(frac=1)
X=df.iloc[:,:20]
y=df['label']

#lets convert the labels into unique integer
from sklearn.preprocessing import LabelEncoder
lbl=LabelEncoder()
y=lbl.fit_transform(y)
y
#male convert to 1
#female convert to 0

#split the dataset into train and test
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
acc_scores=[]
roc_scores=[]
clf=RandomForestClassifier(n_estimators=150)
clf.fit(X_train,y_train)
clf.score(X_train,y_train)
y_pred=clf.predict(X_test)
acc_scores.append(accuracy_score(y_test,y_pred))
roc_scores.append(roc_auc_score(y_test,y_pred))
acc_scores[0],roc_scores[0]

import pickle
pickle.dump(clf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0.077315503,0.083829421,0.036718459,0.008701057,0.131908017,0.123206961,30.75715458,1024.927705,0.846389092,0.478904979,0,0.077315503,0.098706262,0.015655577,0.271186441,0.007990057,0.0078125,0.015625,0.0078125,0.046511628]]))


