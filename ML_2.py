
# coding: utf-8

# In[13]:

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree


# In[20]:

iris=load_iris()
test_idx=[0,50,100] #indexes of data we want to test


# In[21]:

#print(iris.feature_names)


# In[22]:

#print(iris.target_names)


# In[23]:

#print(iris.data[1])
#print(iris.target[1])


# In[24]:

#for i in range(len(iris.target)):
    #print("Example %d: label %s features %s" % (i, iris.target[i], iris.data[i]))
    


# In[26]:

#Training Data
train_target=np.delete(iris.target,test_idx)
train_data=np.delete(iris.data,test_idx,axis=0)


# In[27]:

#Testing Data
test_target=iris.target[test_idx]
test_data=iris.data[test_idx]


# In[29]:

clf = tree.DecisionTreeClassifier()


# In[31]:

clf.fit(train_data,train_target)


# In[32]:

print(test_target)


# In[33]:

print(clf.predict(test_data)) #Prediction


# In[ ]:



