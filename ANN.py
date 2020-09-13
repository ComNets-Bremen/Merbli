#part1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score, recall_score, classification_report

# Importing the dataset
dataset = pd.read_csv('data_set.csv')
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
conditions_1 =[(df['f_1_3'] <=200),
               (df['f_1_3'] <= 300) & (df['f_1_3'] > 200),
               (df['f_1_3'] <= 400) & (df['f_1_3'] > 300),
               (df['f_1_3'] <= 500) & (df['f_1_3'] > 400),
               (df['f_1_3'] <= 600) & (df['f_1_3'] > 500),
               (df['f_1_3'] <= 700) & (df['f_1_3'] > 600),
               (df['f_1_3'] <= 800) & (df['f_1_3'] > 700),
               (df['f_1_3'] <= 3000) & (df['f_1_3'] > 800),]
choices_1 = ['1','2','3','4','5','6','7','8']
df['f_1_3_bin'] = np.select(conditions_1, choices_1, default='0')

#bins for f_5
conditions_2 = [(df['f_1_5'] <=200),
                (df['f_1_5'] <= 300) & (df['f_1_5'] > 200),
                (df['f_1_5'] <= 400) & (df['f_1_5'] > 300),
                (df['f_1_5'] <= 500) & (df['f_1_5'] > 400),
                (df['f_1_5'] <= 600) & (df['f_1_5'] > 500),
                (df['f_1_5'] <= 700) & (df['f_1_5'] > 600),
                (df['f_1_5'] <= 800) & (df['f_1_5'] > 700),
                (df['f_1_5'] <= 3000) & (df['f_1_5'] > 800),]
choices_2 = ['1','2','3','4','5','6','7','8']
df['f_1_5_bin'] = np.select(conditions_2, choices_2, default='0')

#bins for f_7
conditions_3 = [(df['f_1_7'] <=200),
                (df['f_1_7'] <= 300) & (df['f_1_7'] > 200),
                (df['f_1_7'] <= 400) & (df['f_1_7'] > 300),
                (df['f_1_7'] <= 500) & (df['f_1_7'] > 400),
                (df['f_1_7'] <= 600) & (df['f_1_7'] > 500),
                (df['f_1_7'] <= 700) & (df['f_1_7'] > 600),
                (df['f_1_7'] <= 800) & (df['f_1_7'] > 700),
                (df['f_1_7'] <= 3000) & (df['f_1_7'] > 800),]
choices_3 = ['1','2','3','4','5','6','7','8']
df['f_1_7_bin'] = np.select(conditions_3, choices_3, default='0')


#bins for f_9
conditions_4 = [(df['f_1_9'] <=200),
                (df['f_1_9'] <= 300) & (df['f_1_9'] > 200),
                (df['f_1_9'] <= 400) & (df['f_1_9'] > 300),
                (df['f_1_9'] <= 500) & (df['f_1_9'] > 400),
                (df['f_1_9'] <= 600) & (df['f_1_9'] > 500),
                (df['f_1_9'] <= 700) & (df['f_1_9'] > 600),
                (df['f_1_9'] <= 800) & (df['f_1_9'] > 700),
                (df['f_1_9'] <= 3000) & (df['f_1_9'] > 800),]
choices_4 = ['1','2','3','4','5','6','7','8']
df['f_1_9_bin'] = np.select(conditions_4, choices_4, default='0')


#defining input and output
X = dataset.iloc[:, [1,3,4,9,20,21,22,23,25,26,27,28,29]].values  #SHTM 
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


#oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE('minority')
X_train,y_train= smote.fit_sample(X_train,y_train)
X_train,y_train= smote.fit_sample(X_train,y_train)
X_train,y_train= smote.fit_sample(X_train,y_train)
X_train,y_train= smote.fit_sample(X_train,y_train)


#Part 2 - ANN

#Importing the Keras Libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.constraints import maxnorm

#Initializing ANN
classifier = Sequential() #No need to put any arguments as we are going to add layers

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 9, init = 'uniform',activation='relu',kernel_constraint=maxnorm(3), input_dim = 13)) 
classifier.add(Dropout(0.3))

#Adding the second hidden layer
classifier.add(Dense(output_dim = 9, init = 'uniform',kernel_constraint=maxnorm(3),activation='relu'))
classifier.add(Dropout(0.2))

#Adding the third hidden layer
classifier.add(Dense(output_dim = 9, init = 'uniform',kernel_constraint=maxnorm(3),activation='relu'))
classifier.add(Dropout(0.2))

#Adding the fourth hidden layer
classifier.add(Dense(output_dim = 9, init = 'uniform',kernel_constraint=maxnorm(3),activation='relu'))
classifier.add(Dropout(0.1))

#Adding the output layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation='softmax')) 
                              
#Compiling the ANN
#opt = SGD(lr=0.01, momentum=0.9,decay=0.0001) 
#opt = keras.optimizers.RMSprop(learning_rate=0.001)   
classifier.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

#Early Stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

#Checkpoint
mc = ModelCheckpoint('best_model_1.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

#Fitting the ANN to the training set
history=classifier.fit(X_train, y_train, validation_data=(X_test, y_test), epochs =1000, callbacks=[es,mc])


#part3 - Making the predictionbs and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_new = np.argmax(y_pred,axis=1)

print('Precision score: ' + str(precision_score(y_test, y_pred_new, average='micro')))
print('Accuracy score: ' + str(accuracy_score(y_test, y_pred_new)))
print('Recall score: ' + str(recall_score(y_test, y_pred_new, average='micro')))
print(classification_report(y_test, y_pred_new))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_new)

_,train_acc = classifier.evaluate(X_train,y_train)
_,test_acc= classifier.evaluate(X_test,y_test)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

'''from keras.models import load_model
saved_model = load_model('best_model_8.h5')

_,train_acc_1 = saved_model.evaluate(X_train,y_train)
_,test_acc_1 = saved_model.evaluate(X_test,y_test)
print('Train: %.3f, Test: %.3f' % (train_acc_1, test_acc_1))'''

plt.figure()
plt.subplot(211)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend(bbox_to_anchor=(1.15,1),loc="upper right", fontsize ='small')
plt.title('Accuracy')
plt.show()

plt.subplot(212)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend(bbox_to_anchor=(1.15,1),loc="upper right", fontsize ='small')
plt.title('Loss')
plt.show()

#k-fold cross validation
#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = classifier, X = X_train, y= y_train, scoring = "hamming", cv = 10) #cv parameter is the number of folds we need to splitt the data
#accuracies.mean()
#accuracies.std()

'''from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
estimator = history(epochs=200, batch_size=64, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))'''

 



