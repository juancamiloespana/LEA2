####paquetes #####
import seaborn as sns
from sklearn import cluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

#### cargar y escalar datos
iris=sns.load_dataset('iris')
feat=iris[['sepal_length','sepal_width']]
feat.shape
feat_sc=StandardScaler().fit_transform(feat)

db_clust= cluster.DBSCAN()
db_clust.fit(feat_sc)
db_clust.labels_

sns.scatterplot(x=feat_sc[:,0], y=feat_sc[:,1], 
                hue=db_clust.labels_, palette='viridis')


###### definir eps vecinos más cercanos ####

knn=NearestNeighbors(n_neighbors=40)
knn.fit(feat_sc)
distance, *_=knn.kneighbors(feat_sc)
distance_prom=np.mean(distance, axis=0)


###probar dbscan eps =0.51127074 k=9 vecino promedio
## min_sample= 9

db_clust= cluster.DBSCAN(eps=0.41127074, min_samples=9)
db_clust.fit(feat_sc)
db_clust.labels_

sns.scatterplot(x=feat_sc[:,0], y=feat_sc[:,1], 
                hue=db_clust.labels_, palette='viridis')

#####regla del codo para epsilon

from kneed import KneeLocator ## para regla del codo


sns.lineplot(x=np.arange(0,40), y=distance_prom)
kl=KneeLocator(x=np.arange(0,40), y=distance_prom,curve="concave", direction= 'increasing' )
kl.elbow
eps=distance_prom[11]


#####para min sample basado en el espilon identificado hacer for

sil=[] ## para guardar silhouette score

for ms in range(2, 25):
    db=cluster.DBSCAN(eps=eps, min_samples=ms)
    db.fit(feat_sc)
    ss=silhouette_score(feat_sc, db.labels_)
    sil.append(ss)
    

sns.lineplot(x=np.arange(2,25), y=sil, palette='viridis')



####
db_final=cluster.DBSCAN(eps=0.3, min_samples=11)
db_final.fit(feat_sc)
db_final.labels_

sns.scatterplot(x=feat_sc[:,0], y=feat_sc[:,1], 
                hue=db_final.labels_, palette='viridis')




