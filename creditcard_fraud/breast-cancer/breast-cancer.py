import pandas as pd
import numpy as np
import km
from sklearn import ensemble

# For data we use the Wisconsin Breast Cancer Dataset
# Via: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
df = pd.read_csv("data.csv")
feature_names = [c for c in df.columns if c not in ["id", "diagnosis"]]
df["diagnosis"] = df["diagnosis"].apply(lambda x: 1 if x == "M" else 0)
X = np.array(df[feature_names].fillna(0)) # quick and dirty imputation
y = np.array(df["diagnosis"])

# We create a custom 1-D lens with Isolation Forest
model = ensemble.IsolationForest(random_state=1729)
model.fit(X)
lens1 = model.decision_function(X).reshape((X.shape[0], 1))

# We create another 1-D lens with L2-norm
mapper = km.KeplerMapper(verbose=3)
lens2 = mapper.fit_transform(X, projection="l2norm")

# Combine both lenses to create a 2-D [Isolation Forest, L^2-Norm] lens
lens = np.c_[lens1, lens2]

# Create the simplicial complex
graph = mapper.map(lens, 
                   X, 
                   nr_cubes=15, 
                   overlap_perc=0.7, 
                   clusterer=km.cluster.KMeans(n_clusters=2, 
                                               random_state=1618033))

# Visualization
mapper.visualize(graph, 
                 path_html="breast-cancer.html", 
                 title="Wisconsin Breast Cancer Dataset", 
                 custom_tooltips=y, 
                 color_function="average_signal_cluster")