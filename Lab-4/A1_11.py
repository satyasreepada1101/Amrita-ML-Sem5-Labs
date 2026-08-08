import pandas as pd
import numpy as np

# Read the Excel file
data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

# Select only numeric columns
numeric_data = data.select_dtypes(include=["number"])

# Convert to NumPy array
data = numeric_data.values


# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    distance = 0

    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2

    return distance ** 0.5


# Function to assign clusters
def assign_clusters(data, centroids):
    labels = []

    for point in data:
        min_distance = euclidean_distance(point, centroids[0])
        cluster = 0

        for i in range(1, len(centroids)):
            dist = euclidean_distance(point, centroids[i])

            if dist < min_distance:
                min_distance = dist
                cluster = i

        labels.append(cluster)

    return labels


# Function to update centroids
def update_centroids(data, labels, k):
    centroids = []

    for i in range(k):
        cluster_points = []

        for j in range(len(data)):
            if labels[j] == i:
                cluster_points.append(data[j])

        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
        else:
            centroid = data[np.random.randint(len(data))]

        centroids.append(centroid)

    return np.array(centroids)


# Function to perform K-Means
def kmeans(data, k, max_iterations=100):
    # Select initial centroids
    centroids = data[:k].copy()

    for _ in range(max_iterations):
        labels = assign_clusters(data, centroids)

        new_centroids = update_centroids(data, labels, k)

        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, labels


# Run K-Means
k = 3
centroids, labels = kmeans(data, k)

# Print final centroids
print("Final Centroids:")
print(centroids)

# Print cluster assignment
print("\nCluster Assignment:")
for i in range(len(labels)):
    print("Data Point", i + 1, "-> Cluster", labels[i])

# Print number of points in each cluster
print("\nNumber of Points in Each Cluster:")
for i in range(k):
    count = labels.count(i)
    print("Cluster", i, ":", count)