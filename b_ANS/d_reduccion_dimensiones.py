import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#### para pca

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from kneed import KneeLocator ## para analizar número de componentes

iris=sns.load_dataset('iris')
iris2=iris.iloc[:,0:4] ## para eliminar variable especie
sc=StandardScaler().fit(iris2)
feat_sc=sc.transform(iris2)

pca=PCA(n_components=4) ## se puede dar valor de componentes o varianza explicada
pca.fit(feat_sc)


pca.components_ ## lambdas, vectores propios pesos de observadas sobre latentes
pca.explained_variance_ ## valores propios alpha, cuánta varianza es explicada
ve=pca.explained_variance_ratio_ ### procentaje de variable explicada por cada componente

l = pca.transform(feat_sc) ## variables latentes
l[0] ## variables latentes para primera fila

pd.DataFrame(l)

l_manual=np.dot(feat_sc,pca.components_.T ,) ## calcula los l manualmente
l_manual[0]



#### analizar número de componentes con regla del codo
sns.lineplot(x=np.arange(1,5), y=np.cumsum(ve), palette="viridis")


kl=KneeLocator(x=np.arange(1,5), y=np.cumsum(ve),curve="concave", direction="increasing")
kl.elbow

l_sel=l[:,0:2]


####3 para graficar los datos originales de iris con la transformación

sns.scatterplot(x=l_sel[:,0], y=l_sel[:,1], hue=iris['species'])



#####Análisis factorial #########



fa=FactorAnalyzer(n_factors=4, rotation=None)
fa.fit(feat_sc)



w=fa.loadings_ ## WW peso de variables latentes sobre las x
l=fa.transform(feat_sc) ### variables latentes
l[0]
pd.DataFrame(l)
lamb=fa.weights_ ## peso de variables observadas sobre las latentes

ve=fa.get_factor_variance() ## varianza explicada, porcentaje de varianza explicada, porcentaje de varianza explicado acumulado
sns.scatterplot(x=l[:,0], y=l[:,1], hue=iris['species'])