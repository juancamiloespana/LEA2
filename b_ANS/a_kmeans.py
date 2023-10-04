import seaborn as sns ### para los datos y para gráficar
import matplotlib.pyplot as plt
from sklearn import cluster ### modelos de clúster
import pandas as pd
import numpy as np




df_iris = sns.load_dataset('iris')
df_iris.info()

features=df_iris[['sepal_length','sepal_width']]

k=3 ### por contexto del problema

kmedias=cluster.KMeans(n_clusters=k) ## crea el modelo

kmedias.fit(features) ## ajusta modelo a datos


cluster_label= kmedias.labels_ ##numeros de los cluster
df_iris['cluster']= cluster_label ## agregar clusters a dataframe
centroides=kmedias.cluster_centers_ ## centroides



### grafica de cluster y centroides
sns.scatterplot(x='sepal_length', y ='sepal_width', hue='cluster',data=df_iris, palette="viridis")
plt.scatter(x=centroides[:,0], y = centroides[:,1], marker='o', s=200)
plt.show()

