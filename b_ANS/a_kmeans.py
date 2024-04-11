
import seaborn as sns ### para los datos y para gráficar
import matplotlib.pyplot as plt
from sklearn import cluster ### modelos de clúster
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score ## indicador para consistencia de clúster
from kneed import KneeLocator ### para detectar analíticamente el cambio en la pendiente

from clusteval import clusteval ### para detecter numero de cluster automáticamente


df_iris = sns.load_dataset('iris')
df_iris.info()

features=df_iris[['sepal_length','sepal_width']]


### verificar nulos
### imputar
### escalar 

