from sklearn.cluster import KMeans
from sklearn import datasets 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

iris = datasets.load_iris()

X= pd.DataFrame(iris.data)
Y = pd.DataFrame(iris.target)

X.columns = ["Sepal_Length","Sepal_Width","Petal_Length","Petal_Width"]
Y.columns = ["Targets"]

model = KMeans(n_clusters=3)
model.fit(X)

plt.figure(figsize=(14,7))
colormap = np.array(["red","lime","black"])

plt.subplot(1,3,1)
plt.scatter(X.Petal_Length, X.Petal_Width, c = colormap[Y.Targets],s = 40)

plt.title("Real Clusters")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

plt.subplot(1,3,2)
plt.scatter(X.Petal_Length, X.Petal_Width,c = colormap[model.labels_],s = 40)

plt.title("K means Clustering")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns = X.columns)

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=40)
gmm.fit(xs)

plt.subplot(1,3,3)
plt.scatter(X.Petal_Length, X.Petal_Width, c = colormap[0],s = 40)
plt.title("GMM Clustering")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()