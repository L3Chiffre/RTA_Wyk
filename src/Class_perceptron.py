#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 


# In[8]:


iris_data = load_iris()
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)
x = iris_df.drop("species",axis="columns").values
y = iris_df["species"]
y = np.where(y=="setosa",1,-1)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=.3,random_state=1)


# In[24]:


class Perceptron:
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.errors_ = None
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
    
    def predict(self, X):
        return np.where(self._net_input(X) >= 0, 1, -1)
    
    def _net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]


# In[25]:


perceptron = Perceptron(0.01,100)
perceptron.fit(X_train,y_train)
y_pred = perceptron.predict(X_test)
test = pd.DataFrame({"pred":y_pred,
             "test_val":y_test})
perceptron.fit(X_train,y_train)
y_pred = perceptron.predict(X_test)
test = pd.DataFrame({"pred":y_pred,
             "test_val":y_test})
print("acc", (len(test[test["pred"]==test["test_val"]])/len(test))*100,"%"    )


# In[ ]:




