
import seaborn as sns ### para los datos y para gráficar
import matplotlib.pyplot as plt
from sklearn import cluster ### modelos de clúster
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score ## indicador para consistencia de clúster
from kneed import KneeLocator ### para detectar analíticamente el cambio en la pendiente

#from clusteval import clusteval ### para detecter numero de cluster automáticamente
### verificar nulos
### imputar/eliminar
### escalar 

df_iris = sns.load_dataset('iris')
df_iris.info() ## verificar nulos

features=df_iris[['sepal_length','sepal_width']]

sc= StandardScaler().fit(features)
features_sc=sc.transform(features)

###k = 3


kmeans=cluster.KMeans(n_clusters=3, n_init=10)
kmeans.fit(features_sc)

cluster_label=kmeans.labels_
centroides= kmeans.cluster_centers_

sns.scatterplot(x='sepal_length', y='sepal_width', hue=cluster_label, data=df_iris, palette='viridis')

kmeans.inertia_



