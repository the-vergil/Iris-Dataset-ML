#!/usr/bin/env python
# coding: utf-8

# ## Importing important libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


# ### Loading data

# In[2]:


iris = load_iris()
dir(iris)


# ### Converting data into dataframe

# In[5]:


df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df["flower_name"] = df["target"].apply(lambda x: iris.target_names[x])
df.head(3)


# In[7]:


# iris.target_names
# 0 ---> setosa
# 1 ---> versicolor
# 2 ---> virginica


# ### Creating training and testing data

# In[8]:


X = df.drop(["target", "flower_name"], axis=1)
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Logistic Regression Model

# In[15]:


lr = LogisticRegression(max_iter=300)
lr.fit(X_train, y_train)


# In[16]:


lr.score(X_test, y_test)


# In[17]:


lr.predict(X_test)


# In[20]:


np.array(y_test)


# In[26]:


plt.scatter(df["sepal length (cm)"], df["petal length (cm)"], c=df.target)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()


# In[27]:


plt.scatter(df["sepal width (cm)"], df["petal width (cm)"], c=df.target)
plt.xlabel("Sepal width (cm)")
plt.ylabel("Petal width (cm)")
plt.show()


# ## DecisionTreeClassifier

# In[29]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


# In[30]:


dtc.score(X_test,y_test)


# In[32]:


dtc.predict(X_test)


# In[33]:


np.array(y_test)


# ## RandomForest

# In[34]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# In[35]:


rf.score(X_test, y_test)


# In[36]:


rf.predict(X_test)


# In[37]:


np.array(y_test)


# ##### The accuracy of all the three models i.e, LogisticRegression, DecisionTree, RandomForestClassifier is 1.0
