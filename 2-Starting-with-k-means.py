"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="spiral.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
tps1 = time.time()
k=2
model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
model.fit(datanp)
tps2 = time.time() - tps1

print("\nTemps de calcul de Kmeans : ",tps2)
labels = model.labels_
# informations sur le clustering obtenu
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels)

from sklearn.metrics.pairwise import euclidean_distances
dists = euclidean_distances(centroids)
print(dists)


# score_regroupement

from sklearn.cluster import KMeans

cluster_distances = []

for cluster_idx in range(len(centroids)):
    cluster_data = datanp[labels == cluster_idx]
    center = centroids[cluster_idx]

    distances = np.linalg.norm(cluster_data - center, axis=1)

    min_distance = np.min(distances)
    max_distance = np.max(distances)
    mean_distance = np.mean(distances)

    cluster_distances.append({
        "cluster_idx": cluster_idx,
        "min_distance": min_distance,
        "max_distance": max_distance,
        "mean_distance": mean_distance
    })

for cluster_info in cluster_distances:
    print(f"\nCluster {cluster_info['cluster_idx']} - Min Distance: {cluster_info['min_distance']}, Max Distance: {cluster_info['max_distance']}, Mean Distance: {cluster_info['mean_distance']}")


# score_separation


from sklearn.metrics import pairwise_distances, silhouette_score

center_distances = pairwise_distances(centroids)

min_center_distance = np.min(center_distances[np.nonzero(center_distances)])
max_center_distance = np.max(center_distances)
mean_center_distance = np.mean(center_distances[np.nonzero(center_distances)])

print(f"\n Min Distance Between Cluster Centers: {min_center_distance}")
print(f"Max Distance Between Cluster Centers: {max_center_distance}")
print(f"Mean Distance Between Cluster Centers: {mean_center_distance}")

# score_silhouette




score_silhouette = silhouette_score(datanp,labels)

print("\nScore silhouette = ",score_silhouette)


