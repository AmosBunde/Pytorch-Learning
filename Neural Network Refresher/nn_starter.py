import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

"""
Loading data from Kaggle competition
"""

url = "https://kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset"

import pandas as pd
import requests

##data = requests.get('url').content
df = pd.read_csv(url)
df.head()

"""
Separate the independent with dependent variable
"""

import numpy as np


X = np.array(df.loc[ :, df.columns != 'output'])

y = np.array(df['output'])

print(f"X: {X.shape}, y: {y.shape}")




#%% Train / Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size =0.2, random_state=123)

# We will use the standard scaler to scale the data

scaler = StandardScaler()
X_train_scale = scalar.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)


#%% Creating a neural network class

class NeuralNetwork:

    """
    Creating the Neural network constructor
    self: Usual word 
    LR: This is the learning rate for the newural network.
    X_Train: The training dataset for the dependent variable
    y_train: The independent variable used for benching  marking the X_train
    X_test: This is testing dataset without the output variable
    y_test: this is the output for bench marking the 
    """
