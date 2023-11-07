# K-Distance Plot
def plot_k_distance(X, k_range):
    # Fit a k-nearest neighbors model
    k_distances = []
    for k in k_range:
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X)
        distances, _ = nbrs.kneighbors(X)
        max_distances = distances[:, -1]
        k_distances.append(max_distances.mean())

    plt.figure(figsize=(8, 6))
    plt.plot(k_range, k_distances, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Average k-Distance')
    plt.title('K-Distance Plot')
    plt.show()

# Define the range of k values for the k-distance plot
k_range = range(1, 20)

# Plot the k-distance plot
plot_k_distance(X_text, k_range)


# Create a DBSCAN model with the optimal epsilon value
dbscan = DBSCAN(eps=1e-10, min_samples=5)  # You can adjust min_samples as needed

# Fit the DBSCAN model to your text data
dbscan_labels = dbscan.fit_predict(X_text)

# Now you have cluster labels assigned to your data points
dbscan_labels

# column with the clusters obtained by dbscan                                
slice_df['cluster_mapped_dbscan'] = dbscan_labels

plt.title("TF-IDF + DBSCAN Cyberbullying tweets clustering", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=slice_df,
                x=X_embedded[:, 0],
                y=X_embedded[:, 1],
                hue='cluster_mapped_dbscan',
                palette="cubehelix")
#plt.legend(loc='upper right')
plt.legend(bbox_to_anchor=(1.05, 1),
           loc='upper left',
           borderaxespad=1,
           title='Bullying classes')
plt.show()