##basicas
import pandas as pd
import numpy as np


####### redes neuronales

import tensorflow as tf 
from tensorflow import keras

##########
import joblib

url='https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/df_new_flowers.csv'
df_new=pd.read_csv(url)
x=df_new.iloc[:,1:5]


######cargar escalador
sc=joblib.load('salidas\\escalador.joblib')
x_scal=sc.transform(x) ### el escalado de datos neuvos es con base en información de escalado de entrenamiento del modelo

### cargar modelo

modelo=keras.models.load_model('salidas\\modelo.h5')
modelo.summary()

pred_esp=np.argmax(modelo.predict(x_scal), axis=1)
df_new['especie']=pred_esp ### en conjunto de datos nuevos





