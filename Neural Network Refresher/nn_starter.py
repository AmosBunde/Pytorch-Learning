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

    def __init__(self, LR, X_train, y_train, X_test, y_test):
        self.w = np.random.randn(X_train_scale[1])
        self.b = np.random.randn()
        self.LR = LR
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.L_train = []
        self.L_test = []


    def activation(self, x);
        """
        We will use sigmoid function as our activation
        """
        return 1 / (1+np.exp(-x))

    def deactivation(self,x):
        """
        We are showing our sigmoid fucntion for the actiavtion is derived
        """
        return self.activation(x)*(1-self.activation(x))
    

    def forward_func(self, X):
         hidden_0 = np.dot(X, self.w) + self.b
         activate_0 = self.activation(hidden_0)
         return activate_0


    def backward_func(self, X, y_true):
        """
        The objective here is to calculate the gradients
        """
        hidden_0 = np.dot(X, self.w) + self.b
        y_pred = self.forward(X)
        dL_dpred = 2 *(y_pred - y_test)
        dpred_dhidden_0 = self.activation(hidden_0)
        dhidden0_db = 1
        dhidden0_dw = X

        dL_db = dL_dpred * dpred_dhidden_0 * dhidden0_db
        dL_dw = dL_dpred * dpred_dhidden_0 * dhidden0_dw

        return dL_db, dL_dw