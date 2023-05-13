##### cargar paquetes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf ## framework para ajustar redes neuronales
from tensorflow import keras ## módulo tiene la mayoria de funciones para las redes neuronales

##### Cargar y visualizar los datos de mnist

###cargar datos ###
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train.shape
y_train.shape
x_test.shape
y_test.shape

#### visualizar las y ###

np.unique(y_train) ### para ver las categorias que tiene
plt.imshow(x_train[30000],cmap='gray')
plt.show()

######### cargar datos profesor ###
(x_train, y_train), (x_test, y_test) =keras.datasets.fashion_mnist.load_data()

x_train.shape
np.unique(y_train,return_counts=True) ###analizar observaciones por categoria

plt.imshow(x_train[6000], cmap='gray')
y_train[6000]

##############ajustar red neuronal y RF ###################
##########################################################

x_train2 = x_train/255 ## para escalar variables y queden de 0 a 1 (eficiencia)
x_test2=x_test/255

f=28 ## filas de la imagen
c=28 ## columnas de la imagen
fxc=f*c

##3 este reshape es para el bosque aleatorio ##
x_train2r=x_train2.reshape(60000,fxc )
x_train2r.shape

######Random forest
rf=RandomForestClassifier(n_estimators=10) ##instanciar el bosque 

#### Red NN ###

#### capas ocultas cualquier funcion de activación:
### tanh es la mejor para capas ocultas
### capa salida depende de variable respuesta, 
### relu : regresión sólo valores positivos 
### sin función de activación:  Regresión puede tomar el label valores negativos
### label: 0, 1:  sigmoid
### label categorias multiples:  softmax

### arquitectura de la red, hiperparámetros  de arquitectura
ann1=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[f,c]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')    
])

### optimizador, hiperparámetros de optimización de la red

ann1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])




rf.fit(x_train2r,y_train)
ann1.fit(x_train2, y_train, epochs=10, validation_data=(x_test2,y_test))