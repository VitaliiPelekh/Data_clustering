import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist


# Генерування даних
def generate_data(N=1000):
    data, _ = make_blobs(n_samples=N, centers=5, random_state=42)
    return data


# Кластеризація K-means
def k_means_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(data)
    return labels, kmeans.cluster_centers_


# Ієрархічна кластеризація
def agglomerative_clustering(data, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(data)
    return labels


# Оцінка якості кластеризації
def clustering_quality(labels, data, centroids=None, metric='euclidean'):
    if centroids is not None:
        distances = np.min(cdist(data, centroids, metric), axis=1)
    else:
        distances = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            cluster_points = data[labels == labels[i]]
            other_points = cluster_points[np.all(cluster_points != data[i], axis=1)]
            distances[i] = np.min(cdist(data[i].reshape(1, -1), other_points, metric))
    score = silhouette_score(data, labels, metric=metric)
    return distances.mean(), score


def compute_agglomerative_centroids(data, labels, n_clusters):
    centroids = np.zeros((n_clusters, data.shape[1]))
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        centroids[i] = np.mean(cluster_points, axis=0)
    return centroids


# Візуалізація результатів
def visualize_clusters(data, labels, centroids=None, title=""):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')

    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label='Центроїди')
        plt.legend()

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def display_cluster_info(data, labels, centroids):
    for i, center in enumerate(centroids):
        cluster_points = data[labels == i]
        print(f"Cluster {i + 1}:")
        print(f"  Center at: {center}")
        print(f"  Points in cluster: {len(cluster_points)}")


data = generate_data(N=1000)

# K-means
kmeans_labels, kmeans_centroids = k_means_clustering(data, 5)
kmeans_distance, kmeans_score = clustering_quality(kmeans_labels, data, kmeans_centroids)

# Ієрархічна кластеризація
agglomerative_labels = agglomerative_clustering(data, 5)
agglo_distance, agglo_score = clustering_quality(agglomerative_labels, data)
agglo_centroids = compute_agglomerative_centroids(data, agglomerative_labels, 5)

visualize_clusters(data, kmeans_labels, kmeans_centroids, "K-means кластеризація")
visualize_clusters(data, agglomerative_labels, title="Ієрархічна кластеризація")

print("K-means:")
print("Середня відстань до центроїдів: {:.4f}".format(kmeans_distance))
display_cluster_info(data, kmeans_labels, kmeans_centroids)

print("\nІєрархічна кластеризація:")
print("Середня відстань до найближчого кластера: {:.4f}".format(agglo_distance))
display_cluster_info(data, agglomerative_labels, agglo_centroids)
