# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 12:27:50 2020

@author: Shashini
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pickle import dump
from sklearn.metrics import accuracy_score,precision_score, recall_score, classification_report

# Importing the dataset
dataset = pd.read_csv('Dataset_1.csv')
df = pd.DataFrame(dataset)

#bins for f_1
conditions = [(df['f_1_1'] <=200),
              (df['f_1_1'] <= 300) & (df['f_1_1'] > 200),
              (df['f_1_1'] <= 400) & (df['f_1_1'] > 300),
              (df['f_1_1'] <= 500) & (df['f_1_1'] > 400),
              (df['f_1_1'] <= 600) & (df['f_1_1'] > 500),
              (df['f_1_1'] <= 700) & (df['f_1_1'] > 600),
              (df['f_1_1'] <= 800) & (df['f_1_1'] > 700),
              (df['f_1_1'] <= 3000) & (df['f_1_1'] > 800),]
choices = ['1','2','3','4','5','6','7','8']
df['f_1_1_bin'] = np.select(conditions, choices, default='0')

#bins for f_3
conditions_2 =[(df['f_1_3'] <=200),
               (df['f_1_3'] <= 300) & (df['f_1_3'] > 200),
               (df['f_1_3'] <= 400) & (df['f_1_3'] > 300),
               (df['f_1_3'] <= 500) & (df['f_1_3'] > 400),
               (df['f_1_3'] <= 600) & (df['f_1_3'] > 500),
               (df['f_1_3'] <= 700) & (df['f_1_3'] > 600),
               (df['f_1_3'] <= 800) & (df['f_1_3'] > 700),
               (df['f_1_3'] <= 3000) & (df['f_1_3'] > 800),]
choices_2 = ['1','2','3','4','5','6','7','8']
df['f_1_3_bin'] = np.select(conditions_2, choices_2, default='0')


#bins for f_5
conditions_4 = [(df['f_1_5'] <=200),
                (df['f_1_5'] <= 300) & (df['f_1_5'] > 200),
                (df['f_1_5'] <= 400) & (df['f_1_5'] > 300),
                (df['f_1_5'] <= 500) & (df['f_1_5'] > 400),
                (df['f_1_5'] <= 600) & (df['f_1_5'] > 500),
                (df['f_1_5'] <= 700) & (df['f_1_5'] > 600),
                (df['f_1_5'] <= 800) & (df['f_1_5'] > 700),
                (df['f_1_5'] <= 3000) & (df['f_1_5'] > 800),]
choices_4 = ['1','2','3','4','5','6','7','8']
df['f_1_5_bin'] = np.select(conditions_4, choices_4, default='0')


#bins for f_7
conditions_6 = [(df['f_1_7'] <=200),
                (df['f_1_7'] <= 300) & (df['f_1_7'] > 200),
                (df['f_1_7'] <= 400) & (df['f_1_7'] > 300),
                (df['f_1_7'] <= 500) & (df['f_1_7'] > 400),
                (df['f_1_7'] <= 600) & (df['f_1_7'] > 500),
                (df['f_1_7'] <= 700) & (df['f_1_7'] > 600),
                (df['f_1_7'] <= 800) & (df['f_1_7'] > 700),
                (df['f_1_7'] <= 3000) & (df['f_1_7'] > 800),]
choices_6 = ['1','2','3','4','5','6','7','8']
df['f_1_7_bin'] = np.select(conditions_6, choices_6, default='0')


#bins for f_9
conditions_8 = [(df['f_1_9'] <=200),
                (df['f_1_9'] <= 300) & (df['f_1_9'] > 200),
                (df['f_1_9'] <= 400) & (df['f_1_9'] > 300),
                (df['f_1_9'] <= 500) & (df['f_1_9'] > 400),
                (df['f_1_9'] <= 600) & (df['f_1_9'] > 500),
                (df['f_1_9'] <= 700) & (df['f_1_9'] > 600),
                (df['f_1_9'] <= 800) & (df['f_1_9'] > 700),
                (df['f_1_9'] <= 3000) & (df['f_1_9'] > 800),]
choices_8 = ['1','2','3','4','5','6','7','8']
df['f_1_9_bin'] = np.select(conditions_8, choices_8, default='0')


#defining input and output
X = dataset.iloc[:, [1,3,4,9,20,21,22,23,25,26,27,28,29]].values  #Sound,DHT11 and Motion sensors as inputs 
#X = dataset.iloc[:, [1,3,4,20,21,22,23,25,26,27,28,29]].values  #Sound and DHT11 sensors as inputs 
y = dataset.iloc[:, 24].values


# Encoding categorical Data
from sklearn.preprocessing import LabelEncoder
labelencoder_Y= LabelEncoder()
y= labelencoder_Y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
dump(sc, open('scaler_new.pkl', 'wb'))        #saving model scaling parameters


#oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE('minority')
X_train,y_train= smote.fit_sample(X_train,y_train)
X_train,y_train= smote.fit_sample(X_train,y_train)
X_train,y_train= smote.fit_sample(X_train,y_train)
X_train,y_train= smote.fit_sample(X_train,y_train)


# Defining the classifier
#uncomment according to the classifier needed

#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy',max_depth=10,min_samples_leaf=10, max_features=4,random_state = 0)


#Decision Tree
#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth=10,min_samples_leaf = 1, random_state =0)


#KNN
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors =50, metric ='minkowski' , p=2)

#KSVM
#from sklearn.svm import SVC
#classifier = SVC(C= 100, kernel = 'rbf', random_state = 0) #penalty parameter of the error term in SVC which is a regularization term to prevent overfitting, Default was 1. higher the value higher the prevention from overfitting but if it's too high we go for under fitting


#Naive Bayes
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()


#fitting the classifier
classifier.fit(X_train, y_train)

#saving the model
filename = 'finalized_model_new.sav'
pickle.dump(classifier, open(filename, 'wb'))

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print('Precision score: ' + str(precision_score(y_test, y_pred, average='micro')))
print('Accuracy score: ' + str(accuracy_score(y_test, y_pred)))
print('Recall score: ' + str(recall_score(y_test, y_pred, average='micro')))
print(classification_report(y_test, y_pred))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y= y_train, cv = 10) #cv parameter is the number of folds we need to splitt the data
m= accuracies.mean()
print(m)
s= accuracies.std()
print(s)

#calculating train score and test score
train_scores = [classifier.score(X_train, y_train)]
test_scores = [classifier.score(X_test, y_test)]


