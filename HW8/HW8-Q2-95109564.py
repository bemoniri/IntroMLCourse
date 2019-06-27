
import random
import math
from matplotlib import pyplot as plt
import csv 
import numpy as np
import seaborn as sn

# Euclidian Distance between two d-dimensional points
def eucldist(p,q):
    dist = 0.0
    for i in range(0,len(p)):
        dist = dist + (p[i] - q[i])**2
    return math.sqrt(dist)
    
def kmeans(k, datapoints, Max_Iterations):

    d = len(datapoints[0]) 
    i = 0
    cluster = [0] * len(datapoints)
    prev_cluster = [-1] * len(datapoints)
    
    cluster_centers = []
    for i in range(0,k):
        new_cluster = []
        cluster_centers += [random.choice(datapoints)]
    while (cluster != prev_cluster) or (i > Max_Iterations) :
        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1
        for p in range(0,len(datapoints)):
            min_dist = float("inf")
            for c in range(0,len(cluster_centers)):
                dist = eucldist(datapoints[p],cluster_centers[c])
                if (dist < min_dist):
                    min_dist = dist  
                    cluster[p] = c
        for k in range(0,len(cluster_centers)):
            new_center = [0] * d
            members = 0
            for p in range(0,len(datapoints)):
                if (cluster[p] == k):
                    for j in range(0,d):
                        new_center[j] += datapoints[p][j]
                    members += 1
            for j in range(0,d):
                if members != 0:
                    new_center[j] = new_center[j] / float(members)                     
                else: 
                    new_center = random.choice(datapoints)   
            cluster_centers[k] = new_center
    return cluster
    
    


# In[79]:


reader = csv.reader(open("iris.csv", "r"), delimiter=",")
x = list(reader)
result = np.array(x)

X = result[1:,0:-1].astype('float')
y = result[1:,4].astype('float')

y_pred = kmeans(3,X, 100000) 

plt.figure(figsize=(13,13))
plt.subplot(6,2,1)
plt.title('Clustered Data')
plt.scatter(X[:,0],X[:,1], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(6,2,2)
plt.title('Labels')
plt.scatter(X[:,0],X[:,1], c=y, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(6,2,3)
plt.scatter(X[:,0],X[:,2], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(6,2,4)
plt.scatter(X[:,0],X[:,2], c=y, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(6,2,5)
plt.scatter(X[:,0],X[:,3], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(6,2,6)
plt.scatter(X[:,0],X[:,3], c=y, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(6,2,7)
plt.scatter(X[:,1],X[:,2], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(6,2,8)
plt.scatter(X[:,1],X[:,2], c=y, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(6,2,9)
plt.scatter(X[:,1],X[:,3], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(6,2,10)
plt.scatter(X[:,1],X[:,3], c=y, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(6,2,11)
plt.scatter(X[:,2],X[:,3], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(6,2,12)
plt.scatter(X[:,2],X[:,3], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.show()


# In[13]:


reader = csv.reader(open("iris.csv", "r"), delimiter=",")
x = list(reader)
result = np.array(x)

X = result[1:,1:-1].astype('float')
y = result[1:,4].astype('float')

y_pred = kmeans(3,X, 100000) 

plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
plt.title('Clustered Data')
plt.scatter(X[:,0],X[:,1], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,2)
plt.title('Labels')
plt.scatter(X[:,0],X[:,1], c=y, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,3)
plt.scatter(X[:,0],X[:,2], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,4)
plt.scatter(X[:,0],X[:,2], c=y, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,5)
plt.scatter(X[:,1],X[:,2], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,6)
plt.scatter(X[:,1],X[:,2], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.show()


# In[14]:


reader = csv.reader(open("iris.csv", "r"), delimiter=",")
x = list(reader)
result = np.array(x)

X = result[1:,[0,2,3]].astype('float')
y = result[1:,4].astype('float')

y_pred = kmeans(3,X, 100000) 

plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
plt.title('Clustered Data')
plt.scatter(X[:,0],X[:,1], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,2)
plt.title('Labels')
plt.scatter(X[:,0],X[:,1], c=y, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,3)
plt.scatter(X[:,0],X[:,2], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,4)
plt.scatter(X[:,0],X[:,2], c=y, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,5)
plt.scatter(X[:,1],X[:,2], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,6)
plt.scatter(X[:,1],X[:,2], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.show()


# In[15]:


reader = csv.reader(open("iris.csv", "r"), delimiter=",")
x = list(reader)
result = np.array(x)

X = result[1:,[0,1,3]].astype('float')
y = result[1:,4].astype('float')

y_pred = kmeans(3,X, 100000) 

plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
plt.title('Clustered Data')
plt.scatter(X[:,0],X[:,1], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,2)
plt.title('Labels')
plt.scatter(X[:,0],X[:,1], c=y, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,3)
plt.scatter(X[:,0],X[:,2], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,4)
plt.scatter(X[:,0],X[:,2], c=y, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,5)
plt.scatter(X[:,1],X[:,2], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,6)
plt.scatter(X[:,1],X[:,2], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.show()


# In[16]:


reader = csv.reader(open("iris.csv", "r"), delimiter=",")
x = list(reader)
result = np.array(x)

X = result[1:,[0,1,2]].astype('float')
y = result[1:,4].astype('float')

y_pred = kmeans(3,X, 100000) 

plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
plt.title('Clustered Data')
plt.scatter(X[:,0],X[:,1], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,2)
plt.title('Labels')
plt.scatter(X[:,0],X[:,1], c=y, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,3)
plt.scatter(X[:,0],X[:,2], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,4)
plt.scatter(X[:,0],X[:,2], c=y, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,5)
plt.scatter(X[:,1],X[:,2], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')

plt.subplot(3,2,6)
plt.scatter(X[:,1],X[:,2], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.show()


# In[32]:


import random
import math
from matplotlib import pyplot as plt
import csv 
import numpy as np

# Euclidian Distance between two d-dimensional points
def eucldist(p,q):
    dist = 0.0
    for i in range(0,len(p)):
        dist = dist + (p[i] - q[i])**2
    return math.sqrt(dist)
    
def k3means(datapoints, Max_Iterations):
    k = 3
    d = len(datapoints[0]) 
    i = 0
    
    cluster = [0] * len(datapoints)
    prev_cluster = [-1] * len(datapoints)
    
    A1 = []
    A2 = []
    A3 = []
    
    cluster_centers = []
    for i in range(0,k):
        new_cluster = []
        cluster_centers += [random.choice(datapoints)]
        
    while (cluster != prev_cluster) or (i > Max_Iterations) :
        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1
    
        for p in range(0,len(datapoints)):
            min_dist = float("inf")
            
            A1.append(cluster_centers[0])
            A2.append(cluster_centers[1])
            A3.append(cluster_centers[2])
            
            
            for c in range(0,len(cluster_centers)):
                
                dist = eucldist(datapoints[p],cluster_centers[c])
                
                if (dist < min_dist):
                    min_dist = dist  
                    cluster[p] = c
        
        for k in range(0,len(cluster_centers)):
            new_center = [0] * d
            members = 0
            for p in range(0,len(datapoints)):
                if (cluster[p] == k):
                    for j in range(0,d):
                        new_center[j] += datapoints[p][j]
                    members += 1
            
            for j in range(0,d):
                if members != 0:
                    new_center[j] = new_center[j] / float(members)                     
                else: 
                    new_center = random.choice(datapoints)

                    
            
            cluster_centers[k] = new_center
            
    return cluster, A1, A2, A3
    
reader = csv.reader(open("iris.csv", "r"), delimiter=",")
x = list(reader)
result = np.array(x)
X = result[1:,[2, 3]].astype('float')
clusters, A1, A2, A3 = k3means(X, 1000) 


H1 = np.asarray(A1)
H2 = np.asarray(A2)
H3 = np.asarray(A3)

plt.figure()
plt.plot(H1[:,0],H1[:,1], color = 'b')
plt.plot(H2[:,0],H2[:,1], color = 'r')
plt.plot(H3[:,0],H3[:,1], color = 'y')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(X[:,0],X[:,1], c=clusters, cmap=plt.cm.Set1, edgecolor='k')
plt.show()
