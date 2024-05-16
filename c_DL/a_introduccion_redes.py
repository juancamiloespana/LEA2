
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


import keras_tuner as kt ### paquete para afinamiento 
####instalar paquete !pip install keras-tuner



### cargamos los datos 
url='https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/iris.csv'
iris_df= pd.read_csv(url)

## verificar nulos
iris_df.info()
iris_df['type'].value_counts()

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


### arquitectura de la red ###

ann1=keras.models.Sequential([
    keras.layers.InputLayer(input_shape=X_tr.shape[1:]),  ## capa de entrada no es necesaria
    keras.layers.Dense(units=512, activation='relu'),
    keras.layers.Dense(units=256, activation='relu'),  ### capa oculta 1, 128 neuronas, función de activación relu
    keras.layers.Dense(units=128, activation='relu'), ## capa oculta 2, 128 neuronas, función de activación relu
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=3,activation='softmax')  # capa de salida , 128 neuronas, función de activación relu
])

ann1.count_params()



#### hiperparámetros de optimización(entrenamiento)
#### compilador ####
l=keras.losses.SparseCategoricalCrossentropy()

m=keras.metrics.SparseCategoricalAccuracy()

ann1.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.01), loss=l, metrics=m)

##### configurar el fit ###
ann1.fit(X_tr, y_tr, epochs=20, validation_data=(X_te, y_te))





#########a

dor=0.3
sr=0.01

l2_r=keras.regularizers.l2(sr)

fa_cap_oculatas='relu'


ann1=keras.models.Sequential([
    keras.layers.InputLayer(input_shape=X_tr.shape[1:]),  ## capa de entrada no es necesaria
    keras.layers.Dense(units=512, activation='relu', kernel_regularizer=l2_r),
    keras.layers.Dropout(dor),
    keras.layers.Dense(units=256, activation='relu',  kernel_regularizer=l2_r),
    keras.layers.Dropout(dor),### capa oculta 1, 128 neuronas, función de activación relu
    keras.layers.Dense(units=128, activation='relu',  kernel_regularizer=l2_r),
    keras.layers.Dropout(dor),## capa oculta 2, 128 neuronas, función de activación relu
    keras.layers.Dense(units=128, activation='relu',  kernel_regularizer=l2_r),
    keras.layers.Dropout(dor),
    keras.layers.Dense(units=3,activation='softmax')  # capa de salida , 128 neuronas, función de activación relu
])




#### hiperparámetros de optimización(entrenamiento)
#### compilador ####
l=keras.losses.SparseCategoricalCrossentropy()
m=keras.metrics.SparseCategoricalAccuracy()
ann1.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.01), loss=l, metrics=m)

##### configurar el fit ###
ann1.fit(X_tr, y_tr, epochs=20, validation_data=(X_te, y_te))
