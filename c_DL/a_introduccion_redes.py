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

### paquete de afinamiento para nn de tensorflor

import keras_tuner as kt

#### paquetes de evaluación de modelos de sklearn

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


### cargamos los datos 
url='https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/iris.csv'

iris_df= pd.read_csv(url)

### separar respuesta y explicativas
y = iris_df['type']
X= iris_df.iloc[:,0:4]

y.value_counts() ### tres categorías 

### escalado de las variables
sc=StandardScaler().fit(X) ## calcular la media y desviacion para hacer el escalado

#### exportar el escalador
import joblib
joblib.dump(sc, "C:\\cod\\LEA2\\c_DL\\sc.joblib") ### exporta objeto


X_sc=sc.transform(X)  ## escalado cob base en variales escladas

## separar entrenamiento evaluación
X_tr, X_te, y_tr, y_te= train_test_split(X_sc, y, test_size=0.2) 

X_tr.shape

##### definier arquitectura de la red neuronal

ann1= keras.models.Sequential([
    keras.layers.Dense(64, input_shape=(4,),activation='relu'),
    keras.layers.Dense(32, activation='tanh'),
    keras.layers.Dense(3, activation='softmax')
])

ann1.get_weights() ### para observar parametros iniciales, tanto W como B

### definir función de perdida y metrica desempeño
loss=keras.losses.SparseCategoricalCrossentropy()
opt= keras.optimizers.Adam(learning_rate=0.01)

### y la métrica
m= keras.metrics.SparseCategoricalAccuracy(name="SCA")

###definir optimizacion y ajuste(entrenamiento)

ann1.compile(optimizer=opt, loss=loss, metrics=m)
ann1.fit(X_tr, y_tr, epochs=10, validation_data=(X_te, y_te))

#### diangostico: Overfitting

### entrenar una red con regularización

###definir hiperparametros de regularización

dr= 0.1 ### porcentaje de neuronas a eliminar
rs= 0.01 ## fuerza de penalización de L2
l2=keras.regularizers.l2(rs) ### instanciar l2

ann2= keras.models.Sequential([
    keras.layers.Dense(64, input_shape=(4,),activation='relu', kernel_regularizer=l2),
    keras.layers.Dropout(rate=dr),
    keras.layers.Dense(32, activation='tanh', kernel_regularizer=l2),
    keras.layers.Dropout(rate=dr),
    keras.layers.Dense(3, activation='softmax')
])

opt= keras.optimizers.Adam(learning_rate=0.01)
ann2.compile(optimizer=opt, loss=loss, metrics=m)
ann2.fit(X_tr, y_tr, epochs=10, validation_data=(X_te, y_te))
###### afinamiento con grilla

### vamos a afinar optimizador a modo de ejemplo de una variable tipo choice, pero con nyestro diagnostico no seria necesario

hp=kt.HyperParameters()

def model_tuning(hp):
    
    dr= hp.Float("DR", min_value=0.05, max_value= 0.2, step=0.05)
    opti=hp.Choice("OPTI", ['adam', 'sgd' ])
    fa=hp.Choice("FA", ["tanh", "sigmoid"])
    
    ann3= keras.models.Sequential([
        keras.layers.Dense(64, input_shape=(4,),activation=fa, kernel_regularizer=l2),
        keras.layers.Dropout(rate=dr),
        keras.layers.Dense(32, activation=fa, kernel_regularizer=l2),
        keras.layers.Dropout(rate=dr),
        keras.layers.Dense(3, activation='softmax')
        ])
    
    if opti=="adam":
        opti2=keras.optimizers.Adam(learning_rate=0.001)
    else:
        opti2=keras.optimizers.SGD(learning_rate=0.001)
        
    ann3.compile(optimizer=opti2, loss=loss, metrics=m)

    return ann3


#### hyper parametros de grilla
search_model=kt.RandomSearch(
    hypermodel=model_tuning, ## nombre de funcion de construccion modelo
    hyperparameters=hp,
    objective=kt.Objective('val_SCA', direction="max"),
    max_trials=10,
    overwrite=True,
    project_name="res_afin"
)

### este es como el fit pero con afinamiento
search_model.search(X_tr, y_tr, epochs=10, validation_data=(X_te, y_te))
search_model.results_summary()

win_model=search_model.get_best_models(1)[0] ### me muestra 1 modelo y escoge posicion 0

win_model.build()
win_model.summary()

#### exportar modelo ganador

joblib.dump(win_model, 'C:\\cod\\LEA2\\c_DL\\win_model.joblib') ### exportar modelo ganador



####### analizar modelo  ganador 

y_pred= np.argmax(win_model.predict(X_te),axis=1)

cm= confusion_matrix(y_te, y_pred)
cm_disp=ConfusionMatrixDisplay(cm)
cm_disp.plot()

print(classification_report(y_te, y_pred))













