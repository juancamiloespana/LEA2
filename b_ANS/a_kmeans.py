
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


###ejercicio identificando analiticamente el k(número de clúster)####

df_iris=sns.load_dataset('iris') ## cargar data frame a utilizar
x=df_iris[['sepal_length','sepal_width']]
x_s=StandardScaler().fit_transform(x) ### se estándarizan las columnas

###identificar el mejork - método del codo y el de silhouette

wcss=[]
sil=[]

for k in range(1,12):
    km=cluster.KMeans(n_clusters=k, n_init=10)
    km.fit(x_s)
    wcss.append(km.inertia_) ##inertia = wcsss
    label=km.labels_
    if k>1:
        sil_avg=silhouette_score(x_s,label )
        sil.append(sil_avg)

sns.lineplot(x=np.arange(1,12), y=wcss, marker="o", palette="viridis")
sns.lineplot(x=np.arange(2,12), y=sil, marker="o", palette="viridis")

kl=KneeLocator(x=np.arange(1,12), y=wcss,curve="convex", direction="decreasing")
kl.elbow

cl=clusteval(cluster="kmeans", evaluate="silhouette")
cl.fit(x_s)
cl.plot()