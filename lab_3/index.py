# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from urllib.request import urlopen
import seaborn as sns

# Download and load the Iris dataset from the UCI Machine Learning Repository
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv(urlopen(url), names=names)

# Drop the 'class' column as we are interested in unsupervised learning here
iris_data = iris_data.drop('class', axis=1)

# Normalize the data using StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(iris_data)

# Determine the optimal number of clusters using the Elbow method
distortions = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(scaled_data)
    distortions.append(kmeans.inertia_)

# Plot the elbow method results
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Distortion')
plt.title('Elbow Method')
plt.savefig('elbow_method.png')
plt.show()

# Apply K-Means clustering on the data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)

# Apply Agglomerative clustering on the data
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(scaled_data)

# Apply DBSCAN clustering on the data
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_data)

# Evaluate the quality of clustering
kmeans_score = silhouette_score(scaled_data, kmeans_labels)
hierarchical_score = silhouette_score(scaled_data, hierarchical_labels)
if len(np.unique(dbscan_labels)) > 1:
    dbscan_score = silhouette_score(scaled_data[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
else:
    dbscan_score = "N/A"

# Create scatter plots for each clustering method
sns.scatterplot(x=iris_data['sepal_length'], y=iris_data['sepal_width'], hue=kmeans_labels, palette="deep").set_title('KMeans Clustering')
plt.savefig('kmeans_clustering.png')
plt.show()

sns.scatterplot(x=iris_data['sepal_length'], y=iris_data['sepal_width'], hue=hierarchical_labels, palette="deep").set_title('Hierarchical Clustering')
plt.savefig('hierarchical_clustering.png')
plt.show()

sns.scatterplot(x=iris_data['sepal_length'], y=iris_data['sepal_width'], hue=dbscan_labels, palette="deep").set_title('DBSCAN Clustering')
plt.savefig('dbscan_clustering.png')
plt.show()

# Write the results to a markdown file
with open('raport.md', 'w') as f:
    f.write(f'# Rezultatele Clusterizării\n\n')
    f.write(f'![Metoda Elbow](elbow_method.png)\n\n')
    f.write(f'K-Means Silhouette Score: {kmeans_score}\n\n')
    f.write(f'![KMeans Clustering](kmeans_clustering.png)\n\n')
    f.write(f'Hierarchical Silhouette Score: {hierarchical_score}\n\n')
    f.write(f'![Hierarchical Clustering](hierarchical_clustering.png)\n\n')
    f.write(f'DBSCAN Silhouette Score: {dbscan_score}\n\n')
    f.write(f'![DBSCAN Clustering](dbscan_clustering.png)\n\n')
