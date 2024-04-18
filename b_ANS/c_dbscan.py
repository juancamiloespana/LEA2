####paquetes #####
import seaborn as sns
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from kneed import KneeLocator
#### cargar y escalar datos
iris=sns.load_dataset('iris')
feat=iris[['sepal_length','sepal_width']]
feat.shape
sc=StandardScaler().fit(feat) ### este objeto se suele exportar para aplicar el algoritmo en datos nuevos
feat_sc=sc.transform(feat)

#### analizar eps y min_sample usando knn


knn=NearestNeighbors(n_neighbors= 50)
knn.fit(feat_sc)
distancia, *_ = knn.kneighbors(feat_sc)
distancia_k_mean=np.mean(distancia, axis=0)

k=np.arange(1, 51)

sns.lineplot(x=k, y =distancia_k_mean)

#### regla del codo para saber con qué vecino más cercano empieza a aumentar mucho la distancia

kl=KneeLocator(x=k,y=distancia_k_mean, curve='concave', direction='increasing')
min_sample=kl.elbow ## min sample 
eps=distancia_k_mean[kl.elbow] ### eps

########
eps2=eps*0.6

db=cluster.DBSCAN(eps=eps2, min_samples=6)
db.fit(feat_sc)

np.unique(db.labels_, return_counts=True)
sns.scatterplot(x=feat_sc[:,0],y=feat_sc[:,1], hue=db.labels_, palette='viridis')



