import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('churn_modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x_1=LabelEncoder()
x[:, 1]=labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2=LabelEncoder()
x[:, 2]=labelencoder_x_1.fit_transform(x[:, 2])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
x=x[:,1:]

