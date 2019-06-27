#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
import csv 
import seaborn as sn
import pandas as pd

df = pd.read_csv('fashion-mnist.csv')

Y= df.loc[:, 'y'].copy()
X = df.drop(['y'],axis = 1).copy()

n = 10 
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(np.array(X.iloc[i,:]).reshape(28, 28))
    plt.gray()
    plt.title('{}'.format((Y[i])), fontsize = 15)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()
plt.close()


# In[63]:


# Seperate Test and Train!
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)


# In[61]:


# Method 1: Linear Soft SVM

print('Starting Linear SVM!...')
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train) 
print('Done!')
print('Prediction on Test Set!...')
y_pred = clf.predict(X_test)
print('Done!')

print('\n')
print('Confusion Matrix:')
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy:')
print(accuracy_score(y_test, y_pred))

df_cm = pd.DataFrame(cm, range(10),range(10))

plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})# font size
plt.show()


# In[67]:


# Method 2: RBF Soft SVM

print('Starting RBF SVM!...')
clf = svm.SVC(C= 1000000, gamma=0.0000002, kernel='rbf')
clf.fit(X_train, y_train) 
print('Done!')
print('Prediction on Test Set!...')
y_pred = clf.predict(X_test)
print('Done!')

print('\n')
print('Confusion Matrix:')
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy:')
print(accuracy_score(y_test, y_pred))

df_cm = pd.DataFrame(cm, range(10),range(10))

plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})# font size
plt.show()


# In[68]:


# Method 3: kNN

print('Starting kNN...')
neigh = KNeighborsClassifier(n_neighbors=6, p=1)
neigh.fit(X_train, y_train)
print('Done!')
print('Prediction on Test Set!...')
y_pred = neigh.predict(X_test)
print('Done!')

print('\n')
print('Confusion Matrix:')
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy:')
print(accuracy_score(y_test, y_pred))

df_cm = pd.DataFrame(cm, range(10),range(10))

plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})# font size
plt.show()


# In[69]:


# Method 3: DT

print('Starting Decision Tree...')

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
print('Done!')
print('Prediction on Test Set!...')
y_pred = clf.predict(X_test)
print('Done!')

print('\n')
print('Confusion Matrix:')
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy:')
print(accuracy_score(y_test, y_pred))

df_cm = pd.DataFrame(cm, range(10),range(10))

plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})# font size
plt.show()


# In[70]:


# Method 3: Neural Network (with Adam)

print('Starting MLP with ADAM Optimization Algorithm...')

clf = MLPClassifier(solver='adam', hidden_layer_sizes=(100, 100), activation='relu')
clf.fit(X_train, y_train)

print('Done!')
print('Prediction on Test Set!...')
y_pred = clf.predict(X_test)
print('Done!')

print('\n')
print('Confusion Matrix:')
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy:')
print(accuracy_score(y_test, y_pred))

df_cm = pd.DataFrame(cm, range(10),range(10))

plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})# font size
plt.show()


plt.figure()
plt.title('Learning Curve with Adam Algorithm')
plt.ylabel('Cost Function (Cross Entropy)')
plt.xlabel('iteration')

plt.plot(clf.loss_curve_)
plt.show()


# In[72]:


# Method 3: Neural Network (with SGD)

print('Starting MLP with SGD Optimization Algorithm...')

clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(100, 100), activation='tanh')
clf.fit(X_train, y_train)

print('Done!')
print('Prediction on Test Set!...')
y_pred = clf.predict(X_test)
print('Done!')

print('\n')
print('Confusion Matrix:')
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy:')
print(accuracy_score(y_test, y_pred))

df_cm = pd.DataFrame(cm, range(10),range(10))

plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})# font size
plt.show()

plt.figure()
plt.title('Learning Curve with SGD Algorithm')
plt.ylabel('Cost Function (Cross Entropy)')
plt.xlabel('iteration')

plt.plot(clf.loss_curve_)
plt.show()


