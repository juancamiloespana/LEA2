
###basicas
import pandas as pd
import numpy as np

##### datos y modelos sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

####### redes neuronales

import tensorflow as tf 
from tensorflow import keras

### cargamos los datos 
url='https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/iris.csv'
iris_df= pd.read_csv(url)

## verificar nulos
iris_df.info()

### separar respuesta y explicativas
y = iris_df['type']
X= iris_df.iloc[:,0:4]

y.value_counts() ### tres categorías 

### escalado de las variables
sc=StandardScaler().fit(X) ## calcular la media y desviacion para hacer el escalado
X_sc=sc.transform(X)  ## escalado cob base en variales escladas

## separar entrenamiento evaluación
X_tr, X_te, y_tr, y_te= train_test_split(X_sc, y, test_size=0.2) 
X_tr.shape

