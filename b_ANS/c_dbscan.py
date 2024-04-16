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


