__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: Scikit-Learn Clustering
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN


# Load data
def load_data():
    '''
    Create data points
    :return:
    '''
    # Create random data points in 4 clusters
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=1000, # number of samples
                      centers=[[1, 1], [2, 2], [3, 4], [5, 5]], # clusters' means
                      cluster_std=0.4, # clusters' std
                      random_state=1)

    return X, y


# Plot K-means clustering
def plot_kmeans(X, kmeans, axes):
    '''
    Plot k-means clustering results
    :param X: data points
    :param kmeans: k-means clustering model
    :param axes:
    :return:
    '''
    labels = kmeans.labels_
    k = len(set(labels))
    centroids = kmeans.cluster_centers_

    # Create colors for the clusters
    colors = plt.get_cmap('Spectral')(np.linspace(0, 1, k))

    for l, color in zip(range(k), colors):
        if l == -1:
            color = 'gray' # color gray for outliers

        labeled = (labels == l)
        cluster_center = centroids[l] # cluster centroid ofdata point

        # Plot clustered data points
        axes.scatter(X[labeled, 0], X[labeled, 1], marker='o', alpha=0.4, color=color)

        # Plot centroids
        axes.scatter(cluster_center[0], cluster_center[1], marker='+', color='k', s=100)

        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title('K-MEANS CLUSTERING (K={})'.format(k), fontsize=12)
    return


# Plot DBSCAN clustering
def plot_dbscan(X, dbscan, axes):
    '''
    Plot dbscan clustering results
    :param X: data points
    :param dbscan: dbscan clustering model
    :param axes:
    :return:
    '''
    labels = dbscan.labels_
    unique_labels = set(labels)

    # Core sample indices (labeled datapoints vs outliers)
    core_sample_indices = np.zeros_like(labels, dtype=bool)
    core_sample_indices[dbscan.core_sample_indices_] = True
    outliers_indices = ~core_sample_indices

    # Create colors for the clusters
    colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(unique_labels)))

    for l, color in zip(unique_labels, colors):
        if l == -1:
            color = 'gray' # color gray for outliers

        labeled = (labels == l)

        # Plot clustered data points
        X_ = X[labeled & core_sample_indices]
        axes.scatter(X_[:, 0], X_[:, 1], marker='o', alpha=0.4, color=color)

        # Plot outliers
        X_ = X[labeled & outliers_indices]
        axes.scatter(X_[:, 0], X_[:, 1], marker='o', alpha=0.5, color=color)

        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title('DBSCAN CLUSTERING', fontsize=12)
    return


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load a sample data
    X, y = load_data()

    # Kmeans clustering
    kmeans_3 = KMeans(init="k-means++", n_clusters=3, n_init=20).fit(X) # k-means clustering with k=3
    kmeans_4 = KMeans(init="k-means++", n_clusters=4, n_init=20).fit(X) # k-means clustering with k=4

    # DBSCAN clustering
    epsilon = 0.3    # specifies how close points should be to each other to be considered a part of a cluster
    min_points = 5   # minimum number of points to form a dense region
    dbscan = DBSCAN(eps=epsilon, min_samples=min_points).fit(X)

    # Plot settings
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(17, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # Plot data
    axes[0, 0].scatter(X[:, 0], X[:, 1], marker='o', edgecolors='k', alpha=0.5, color='gray')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    axes[0, 0].set_title('DATA SAMPLES', fontsize=12)

    # Plot DBSCAN clustering
    plot_dbscan(X, dbscan, axes[0, 1])

    # Plot K-MEANS clustering
    plot_kmeans(X, kmeans_3, axes[1, 0]) # k-means with 3 clusters
    plot_kmeans(X, kmeans_4, axes[1, 1]) # k-means with 4 clusters

    # To save the plot locally
    plt.savefig('sklearn_clustering.png', bbox_inches='tight')
    plt.show()