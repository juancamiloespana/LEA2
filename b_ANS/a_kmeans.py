
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

cluster_label=kmeans.labels_ ### ver los cluster de cada fila
centroides= kmeans.cluster_centers_ ### valores de los centroides

sns.scatterplot(x='sepal_length', y='sepal_width', hue=cluster_label, data=df_iris, palette='viridis')

kmeans.inertia_ ###wcss 
silhouette_score(features_sc,cluster_label) ### silhouette score



#### este for es para calcular el número de cluster de mejor desempeño, ya sea por regla del codo del wcss
### o el coeficiente de silhouette que sea el menor
wcss=[]
ss=[]

for i in range(1,10):
    kmeans=cluster.KMeans(n_clusters=i, n_init=10)
    kmeans.fit(features_sc)
    wcss.append(kmeans.inertia_)
    if i>1:
        ss_i=silhouette_score(features_sc, kmeans.labels_)
        ss.append(ss_i)
    

### análisis gráfico

sns.lineplot(x=range(1,10), y=wcss, marker='o')
plt.title("N_cluster vs WCSS")
plt.show()

    
sns.lineplot(x=range(2,10), y=ss, marker='o')
plt.title("N_cluster vs Silhouette score")
plt.show()


### regla del codo analítica ######

k = KneeLocator(x=range(1,10), y=wcss,curve='convex', direction='decreasing' )
k.elbow ### este es el número de de cluster, valor de x en que cambia la pendiente



