#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initial step:
# import fundamental pkgs
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import datetime as dt

# import visualization pkgs
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML
get_ipython().run_line_magic('matplotlib', 'inline')

# import machine learning pkgs
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc


# In[2]:


# Next step:
# load data and name the dataFrame as df
df = pd.read_csv('data.csv',low_memory=False)


# In[ ]:


# Have a look at data frame
df.head()


# In[ ]:


# Check the number of rows and cols
df.shape


# In[ ]:


# Check the columns' info
list(df.columns)

