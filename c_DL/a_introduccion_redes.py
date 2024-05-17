
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


import joblib



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


########


hp=kt.HyperParameters()


def hyper_mod(hp):
    
    dor=hp.Float("DOR", min_value=0.05, max_value=0.4, step=0.05)
    sr=hp.Float("SR", min_value=0.005, max_value=0.02, step=0.003)
    fa=hp.Choice('FA_CO', ['tanh', 'sigmoid', 'relu'])
    opt=hp.Choice('opt', ['Adam', 'SGD'])
    
    l2_r=keras.regularizers.l2(sr)
    
    
    ann1=keras.models.Sequential([
        keras.layers.InputLayer(input_shape=X_tr.shape[1:]),  ## capa de entrada no es necesaria
        keras.layers.Dense(units=512, activation=fa, kernel_regularizer=l2_r),
        keras.layers.Dropout(dor),
        keras.layers.Dense(units=256, activation=fa,  kernel_regularizer=l2_r),
        keras.layers.Dropout(dor),### capa oculta 1, 128 neuronas, función de activación relu
        keras.layers.Dense(units=128, activation=fa,  kernel_regularizer=l2_r),
        keras.layers.Dropout(dor),## capa oculta 2, 128 neuronas, función de activación relu
        keras.layers.Dense(units=128, activation=fa,  kernel_regularizer=l2_r),
        keras.layers.Dropout(dor),
        keras.layers.Dense(units=3,activation='softmax')  # capa de salida , 128 neuronas, función de activación relu
    ])
    
    l=keras.losses.SparseCategoricalCrossentropy()
    m=keras.metrics.SparseCategoricalAccuracy()
    
    if opt == 'Adam':
        opt1= keras.optimizers.Adam(learning_rate=0.01)
    else:
        opt1= keras.optimizers.SGD(learning_rate=0.01)
        
    
    ann1.compile(optimizer=opt1, loss=l, metrics=m)

    return ann1



search_model = kt.RandomSearch(
    hypermodel= hyper_mod,
    hyperparameters=hp,
    objective=kt.Objective('val_sparse_categorical_accuracy', direction='max'),
    max_trials=5,
    overwrite=True,
    directory="res",
    project_name="afin"
    
)

### esto es equivalente al fit del modelo 
search_model.search(X_tr, y_tr, epochs=10, validation_data=(X_te, y_te))

search_model.results_summary() ### resultados de afinamiento

model_winner=search_model.get_best_models(1)[0]

model_winner.count_params() ### una red neuronal
model_winner.summary() ### una red neuronal

hps_winner= search_model.get_best_hyperparameters(2)[1]
hps_winner.values


################### analizar el modelo #####

pred_test=np.argmax(model_winner.predict(X_te), axis=1)
X_te.shape
print(metrics.classification_report(y_te, pred_test))


pred_tr=np.argmax(model_winner.predict(X_tr), axis=1)
print(metrics.classification_report(y_tr,pred_tr ))


cm=metrics.confusion_matrix(y_te, pred_test)
cm_disp=metrics.ConfusionMatrixDisplay(cm)
cm_disp.plot()


##########

y_total=np.array(y) ### para convertir y en array
X_sc.shape

model=hyper_mod(hps_winner) 
model.fit(X_sc, y_total, epochs=20) ##  queda entrenado el modelo con todos los datos

#### se exportan insumos necesarios para prediccion

###escalador
nombre="hola"
ruta="salidas\\"

nombre_completo= ruta+nombre


joblib.dump(sc, nombre_completo) ### para exporttar objetos de python

model.save('salidas\\modelo.h5') ## para exportar modelo utilizando la función de tensorflow para evitar conflictos



