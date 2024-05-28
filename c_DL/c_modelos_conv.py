#####librerias
import numpy as np

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

##### para afinamiento #########

import keras_tuner as kt

######## cargar datos #####

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train.shape
x_train[1]
x_train.max()
x_train.min()

y_train.shape
np.unique(y_train, return_counts=True)


x_test.shape
y_test.shape


plt.imshow(x_train[1],cmap='gray')
y_train[1]
plt.show()

##### escalar variables #####
x_trains=x_train/255
x_tests=x_test/255

f=x_train.shape[1]
c=x_train.shape[2]
fxc= f*c


#########################################################
######### red convolucional #############################
#########################################################


fa='tanh' ### fucnion de activación para todas las capas

cnn1=keras.models.Sequential([
    keras.layers.Conv2D(filters=16, kernel_size=(3,3),  activation=fa, input_shape=[f,c,1]), ## capa convolucional
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=fa),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation=fa),
    keras.layers.Dense(10, activation="softmax")
    
])

cnn1.summary()


lr=0.002 ## tasa de aprendizaje define si mueve mucho los parámetros o no
optim=keras.optimizers.Adam(lr) ### se configar el optimizador
cnn1.compile(optimizer=optim, loss="sparse_categorical_crossentropy",metrics=['accuracy'] )
cnn1.fit(x_trains, y_train, epochs=5,validation_data=(x_tests, y_test), batch_size=30)

############################################################
#########  Afinamiento de hiperparámetros ##################
###########################################################

hp=kt.HyperParameters()

num_conv_layers=3
def build_model(hp):
    
    ####### definición de hiperparámetros de grilla 
    
    num_conv_layers = hp.Int('num_conv_layers', min_value=1, max_value=3)
    conv_filters = [hp.Int(f'conv_{i+1}_filter', min_value=1, max_value=32, step=16) for i in range(num_conv_layers)]
    conv_kernels = [hp.Choice(f'conv_{i+1}_kernel', values=[3, 1]) for i in range( num_conv_layers)]
    activation = hp.Choice('activation', values=['relu', 'tanh'])
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
    dense_units = hp.Int('dense_units', min_value=8, max_value=64, step=32) 
    
    ####### creación de modelo sequential vacío y capa de entrada

    model = keras.models.Sequential()### se crea modelo sin ninguna capa
    model.add(keras.layers.InputLayer(input_shape=(f, c, 1))) ### se crea capa de entrada
    
    ##### agregar capas convolucionales de acuerdo a hiperparáemtro de capas
    
    for i in range( num_conv_layers):
        model.add(keras.layers.Conv2D(filters=conv_filters[i], kernel_size=(conv_kernels[i], conv_kernels[i]), activation=activation))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    
    ### agregar capas densas siempre estándar al final de la red 
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=dense_units, activation=activation))
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

tuner = kt.RandomSearch(
    build_model,
    hyperparameters=hp,
    objective='val_accuracy',
    max_trials=4,
    directory='my_dir',
    overwrite=True,
    project_name='cnn_tuning'
)

tuner.search(x_train, y_train, epochs=1, validation_data=(x_test, y_test), batch_size=600)

tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model.summary()
best_hps.values


########################################################
########## red neuronal  estándar ######################
#########################################################


fa='tanh' ### fucnion de activación para todas las capas

ann1=keras.models.Sequential([
     keras.layers.Flatten(input_shape=[f,c]),
     keras.layers.Dense(128, activation=fa),
     keras.layers.Dense(32, activation=fa),
     keras.layers.Dense(10, activation="softmax")
 ])

ann1.count_params() ### contar parametros del modelo

lr=0.002 ## tasa de aprendizaje define si mueve mucho los parámetros o no
optim=keras.optimizers.Adam(lr) ### se configar el optimizador

ann1.compile(optimizer=optim, loss="sparse_categorical_crossentropy",metrics=['accuracy'] )
ann1.fit(x_trains, y_train, epochs=10,validation_data=(x_tests, y_test), batch_size=30)
