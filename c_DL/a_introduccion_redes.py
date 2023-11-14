#####paquete básicos ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

#### paquetes de sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#### paquetes de redes neuronales

import tensorflow as tf
from tensorflow import keras


### cargamos los datos 
url='https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/iris.csv'

iris_df= pd.read_csv(url)

### separar respuesta y explicativas
y = iris_df['type']
X= iris_df.iloc[:,0:4]

y.value_counts() ### tres categorías 

### escalado de las variables
X_sc=StandardScaler().fit_transform(X)

## separar entrenamiento evaluación
X_tr, X_te, y_tr, y_te= train_test_split(X_sc, y, test_size=0.2) 


##### definier arquitectura de la red neuronal

ann1= keras.models.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='tanh'),
    keras.layers.Dense(3, activation='softmax')
])

### definir función de perdida y metrica desempeño
loss=keras.losses.SparseCategoricalCrossentropy()
opt= keras.optimizers.Adam(learning_rate=0.01)

### y la métrica
m= keras.metrics.SparseCategoricalAccuracy()


###definir optimizacion y ajuste(entrenamiento)

ann1.compile(optimizer=opt, loss=loss, metrics=m)
ann1.fit(X_tr, y_tr, epochs=10, validation_data=(X_te, y_te))

X_tr.shape
ann1.count_params()
ann1.summary()




