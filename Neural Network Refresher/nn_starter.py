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

