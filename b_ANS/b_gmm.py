
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
sc=StandardScaler().fit(x)
x_sc=sc.transform(x)

sns.scatterplot(x=x_sc[:,0], y=x_sc[:,1])

gmm=mixture.GaussianMixture(n_components=4, covariance_type='full', n_init=5)

gmm.fit(x_sc)

gmm.score(x_sc) ### score por defecto es logaritmo de la función de verosimilitud es adiminesional y no se interpreta por sí solo
gmm.predict_proba(x_sc) ## probabilidad de pertecencer a cada cluster
gmm.predict(x_sc) ### para conocer los label del cluster.

##### afinamiento de hiperparámetros ##########

param_grid={
    'n_components': [2,3, 4, 5, 6, 7],
    'covariance_type':['full', 'diag', 'spherical', 'tied'],
    'n_init': [5, 15, 30]
}

gs=GridSearchCV(gmm, param_grid=param_grid )
gs.fit(x_sc)

gs.cv_results_
gs.best_params_

df_resultados=pd.DataFrame(gs.cv_results_)
pd.set_option('display.max_colwidth',None)
df_resultados.sort_values('mean_test_score', ascending=False)[['params','mean_test_score']]

gmm_win=gs.best_estimator_
label=gmm_win.predict(x_sc)
gmm_win.predict_proba(x_sc)


#### graficar los cluster
sns.scatterplot(x=x_sc[:,0], y=x_sc[:,1], hue=label, palette='viridis')
plt.title("clusters de acuerdo a gmm")
plt.show()