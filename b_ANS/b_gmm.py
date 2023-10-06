
import seaborn as sns ### para los datos y para gráficar
import matplotlib.pyplot as plt
from sklearn import mixture ### modelos de clúster
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score ## indicador para consistencia de clúster
from kneed import KneeLocator ### para detectar analíticamente el cambio en la pendiente
from sklearn.model_selection import GridSearchCV

df_iris=sns.load_dataset('iris')

x=df_iris[['sepal_length','sepal_width']]
x_s=StandardScaler().fit_transform(x)

sns.scatterplot(x=x_s[:,0], y=x_s[:,1])


gmm=mixture.GaussianMixture(n_components=5, covariance_type='full', n_init=5)
mixture.GaussianMixture()
gmm.fit(x_s)
gmm.aic(x_s) ## indicador para comparar con otro modelo por si solo no se interpreta
gmm.bic(x_s)
gmm.score(x_s)
gmm.predict_proba(x_s)[0]


paramg={
    'n_components':[1,2,3,4,5,6,7,8,9,10],
    'covariance_type':['full', 'tied', 'spherical','diag'],
    'n_init':[4,5,6]
}

gs=GridSearchCV(gmm, param_grid=paramg)
gs.fit(x_s)

df_resultados=pd.DataFrame(gs.cv_results_)

df_resultados[['params', 'mean_test_score']].sort_values('mean_test_score', ascending=False)
df_resultados['params'][112]


gmm_gan=gs.best_estimator_
cluster=gmm_gan.fit_predict(x_s)
sns.scatterplot(x=x_s[:,0], y=x_s[:,1], hue=cluster, palette="viridis")