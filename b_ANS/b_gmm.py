
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
