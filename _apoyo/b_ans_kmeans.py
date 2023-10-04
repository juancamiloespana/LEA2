from sklearn import cluster
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


####observar los datos 

iris= sns.load_dataset('iris')
iris.info()
iris.value_counts('species')

features = iris[['sepal_length', 'sepal_width']]

# Specify the number of clusters (K) you want to find
k = 3

# Create a K-Means model
kmeans = cluster.KMeans(n_clusters=k)

# Fit the model to your data
kmeans.fit(features)

# Get cluster labels for each data point
cluster_labels = kmeans.labels_

# Add the cluster labels to the original DataFrame
iris['Cluster'] = cluster_labels

# Print the centroids of each cluster
centroids = kmeans.cluster_centers_
print("Centroids:\n", centroids)

# Plot the data points and cluster centroids
sns.scatterplot(x='sepal_length', y='sepal_width', hue='Cluster', data=iris, palette='viridis')
plt.scatter(x=centroids[:, 0], y=centroids[:, 1], marker='o', s=100, c='red', label='Centroids')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-Means Clustering of Iris Sepal Data')
plt.legend()
plt.show()



sns.scatterplot(x='sepal_length', y='sepal_width', hue='Cluster', data= iris, palette='viridis')
plt.scatter()


wcss=[]
silhouette_scores = []

for k in range(2,15):
    kmeans=cluster.KMeans(k)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
    cluster_labels2=kmeans.labels_
    silhouette_avg=silhouette_score(features, cluster_labels2)
    silhouette_scores.append(silhouette_avg)
    
    

sns.lineplot(x=range(2,15), y =silhouette_scores, marker ="o", color="red")
sns.lineplot(x=range(1,15),y= wcss, marker='o', palette="viridis")

 ########## 
 



import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos de diamantes desde seaborn
diamonds = sns.load_dataset('diamonds')

# Seleccionar las características relevantes para clustering (precio y carat)
features = diamonds[['price', 'carat']]

# Estandarizar las características para que tengan media cero y varianza unitaria
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Especificar el número de clusters (k) que deseas encontrar
k = 3

# Crear un modelo K-Means
kmeans = KMeans(n_clusters=k, random_state=42)

# Ajustar el modelo a los datos estandarizados
kmeans.fit(features_scaled)

# Obtener las etiquetas de cluster asignadas a cada diamante
cluster_labels = kmeans.labels_

# Agregar las etiquetas de cluster al DataFrame original
diamonds['Cluster'] = cluster_labels

# Visualizar los resultados del clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='carat', hue='Cluster', data=diamonds, palette='tab10')
plt.xlabel('Precio')
plt.ylabel('Peso en Quilates (Carat)')
plt.title('Segmentación de Diamantes por Precio y Peso en Quilates')
plt.legend(title='Cluster')
plt.show()