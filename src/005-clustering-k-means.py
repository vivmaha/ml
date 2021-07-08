'''
I'm following along 
https://realpython.com/k-means-clustering-python/

I sprinked some visualizations as I followed along with seaborn
'''

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from kneed import KneeLocator

# Just to make the results deterministic. In practice we won't do this
random_state = 42

features, labels_true = make_blobs(
    n_samples=200,
    centers=3,
    cluster_std=1,
    random_state=random_state
)

# Scale features since k-means depends on distance
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


def visualize_input():
    df = pd.DataFrame(
        {'x': features_scaled[:, 0], 'y': features_scaled[:, 1]}
    )
    sns.scatterplot(data=df, x='x', y='y')
    plt.show()

# visualize_input()
# ^ This showed that the clustering was a good fit for k-means


def get_kmeans(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(features_scaled)
    return kmeans


clusters = []
sse = []
for n_clusters in range(1, 11):
    kmeans = get_kmeans(n_clusters)
    clusters.append(n_clusters)
    sse.append(kmeans.inertia_)


def visualize_elbow():
    df = pd.DataFrame({
        'clusters': clusters,
        'sse': sse,
    })
    sns.lineplot(data=df, x='clusters', y='sse')
    plt.show()

# visualize_elbow()
# ^ This shows that the elbow is at 3


# We visually saw it was 3 above, but we can also programmatically find it
kl = KneeLocator(clusters, sse, curve="convex", direction="decreasing")
n_clusters = kl.elbow

kmeans = get_kmeans(n_clusters)


def visualize_fit():
    df = pd.DataFrame({
        'x': features_scaled[:, 0],
        'y': features_scaled[:, 1],
        # Converting to string to prevent seaborn to using a sequential color
        # palette
        'label': map(str, kmeans.labels_)
    })
    sns.scatterplot(data=df, x='x', y='y', hue='label')
    plt.show()


visualize_fit()
# ^ This showed that the fit worked
