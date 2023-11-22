def calculate_top_words_per_category(data, categories, vectorizer, n_words):
    top_words = {}
    for category in categories:
        # Filter data for each category
        category_data = data[data['cyberbullying_type'] == category]
        # Calculate TF-IDF and get top words
        tfidf_matrix = vectorizer.transform(category_data['cleaned'])
        sum_tfidf = np.sum(tfidf_matrix, axis=0)
        # If it's a sparse matrix, convert to a dense array
        if isinstance(sum_tfidf, np.matrix):
            sum_tfidf = sum_tfidf.A1
        sorted_indices = np.argsort(sum_tfidf)[::-1][:n_words]
        top_category_words = np.array(vectorizer.get_feature_names_out())[sorted_indices]
        top_words[category] = set(top_category_words.tolist())  # Convert array to list before creating a set
    return top_words

def calculate_top_words_per_cluster(X, clusters, vectorizer, n_words):
    top_cluster_words = {}
    for i in range(np.max(clusters) + 1):
        # Find the indices of rows in X that belong to the current cluster
        cluster_indices = np.where(clusters == i)
        # Sum the TF-IDF vectors for each word in the cluster
        sum_tfidf = np.sum(X[cluster_indices], axis=0)
        # Convert to a dense array if needed
        if isinstance(sum_tfidf, np.matrix):
            sum_tfidf = sum_tfidf.A1
        # Get the top word indices sorted by TF-IDF value
        top_indices = np.argsort(sum_tfidf)[::-1][:n_words]
        # Retrieve the top words
        top_words_list = np.array(vectorizer.get_feature_names_out())[top_indices]
        top_cluster_words[i] = set(top_words_list)
    return top_cluster_words

# Retrieve again all top words per category and cluster 
top_words_category = calculate_top_words_per_category(slice_df, slice_df['cyberbullying_type'].unique(), vectorizer, 20)
top_words_cluster_kmeans = calculate_top_words_per_cluster(X_text, slice_df['cluster_kmeans'], vectorizer, 20)
top_words_cluster_spectral = calculate_top_words_per_cluster(X_text, slice_df['cluster_spectral'], vectorizer, 20)
top_words_cluster_agglo = calculate_top_words_per_cluster(X_text, slice_df['cluster_agglo'], vectorizer, 20)

def match_clusters_to_categories(cluster_top_words, category_top_words, unique_categories):
    matches = {}
    used_categories = set()
    cluster_ids = list(cluster_top_words.keys())

    # Sort clusters by the sum of all their similarity scores to prioritize those with higher overall similarity
    cluster_ids.sort(key=lambda cid: sum(
        len(cluster_top_words[cid].intersection(category_top_words[cat])) / len(cluster_top_words[cid].union(category_top_words[cat]))
        for cat in category_top_words
    ), reverse=True)

    while cluster_ids:
        cluster_id = cluster_ids.pop(0)
        cluster_words = cluster_top_words[cluster_id]
        similarity_scores = {}

        # Calculate similarity scores for all categories against this cluster
        for category, category_words in category_top_words.items():
            intersection = cluster_words.intersection(category_words)
            union = cluster_words.union(category_words)
            jaccard_similarity = len(intersection) / len(union)
            similarity_scores[category] = jaccard_similarity

        # Sort categories by similarity score, prioritize unused categories
        sorted_categories = sorted(similarity_scores, key=lambda cat: (cat not in used_categories, similarity_scores[cat]), reverse=True)
        best_match_category = sorted_categories[0]
        matches[cluster_id] = best_match_category
        used_categories.add(best_match_category)

        # If all categories are used, allow reusing categories for remaining clusters
        if len(used_categories) == len(unique_categories):
            used_categories.clear()

    return matches


# Extract the unique categories from your data
unique_categories = np.unique(slice_df['cyberbullying_type'])    
matches_kmeans = match_clusters_to_categories(top_words_cluster_kmeans, top_words_category,unique_categories)
matches_spectral = match_clusters_to_categories(top_words_cluster_spectral, top_words_category,unique_categories)
matches_agglo = match_clusters_to_categories(top_words_cluster_agglo, top_words_category,unique_categories)


# Mapping dictionaries for K-Means and Spectral Clustering
cluster_map_kmeans = {cluster: category for cluster, category in matches_kmeans.items()}
cluster_map_spectral = {cluster: category for cluster, category in matches_spectral.items()}
cluster_map_agglo = {cluster: category for cluster, category in matches_agglo.items()}

print('Cluster mappings for kmeans:')
print(cluster_map_kmeans)

print('Cluster mappings for spectral:')
print(cluster_map_spectral)

print('Cluster mappings for Agglomerative:')
print(cluster_map_agglo)

# Apply the mappings
slice_df['cluster_mapped_kmeans'] = slice_df['cluster_kmeans'].map(cluster_map_kmeans)
slice_df['cluster_mapped_spectral'] = slice_df['cluster_spectral'].map(cluster_map_spectral)
slice_df['cluster_mapped_agglomerative'] = slice_df['cluster_agglo'].map(cluster_map_agglo)


## Aditional tests and checks to evaluate every combination of mapping for each method
def evaluate_mappings(true_labels, predicted_clusters, categories, num_samples=None):
    # Convert to lists for consistent indexing
    true_labels = list(true_labels)
    predicted_clusters = list(predicted_clusters)

    # Generate all unique clusters
    unique_clusters = sorted(set(predicted_clusters))

    # Generate permutations for the number of unique categories
    # and then product combinations if there are more clusters than categories
    if len(unique_clusters) <= len(categories):
        category_combinations = list(permutations(categories, len(unique_clusters)))
    else:
        # First, assign one category to each cluster
        base_permutations = list(permutations(categories))
        # Then, for the remaining clusters, allow categories to repeat
        additional_combinations = list(product(categories, repeat=len(unique_clusters) - len(categories)))
        # Combine the permutations with the additional combinations
        category_combinations = [perm + comb for perm in base_permutations for comb in additional_combinations]

    # If num_samples is specified and less than the total number of combinations, sample a subset
    if num_samples and num_samples < len(category_combinations):
        category_combinations = random.sample(category_combinations, num_samples)

    best_mapping = None
    best_accuracy = 0

    # Test each combination
    for category_comb in category_combinations:
        # Create a mapping for this combination
        mapping = {cluster: category for cluster, category in zip(unique_clusters, category_comb)}
        # Map the predicted clusters to categories using this mapping
        mapped_predictions = [mapping.get(cluster, None) for cluster in predicted_clusters]
        # Calculate the accuracy of this mapping
        accuracy = sum(1 for i, prediction in enumerate(mapped_predictions) if prediction == true_labels[i]) / len(true_labels)
        # If this is the best mapping so far, remember it
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = mapping

    return best_mapping, best_accuracy



# Generate the actual true labels and predicted clusters
actual_true_labels = slice_df['cyberbullying_type'].values
predicted_clusters_kmeans = slice_df['cluster_kmeans'].values
predicted_clusters_spectral = slice_df['cluster_spectral'].values

# Optionally shuffle the data if computational resources are limited
# This is a strategy to get a diverse subset if you can't afford to test all permutations
actual_true_labels, predicted_clusters_kmeans, predicted_clusters_spectral = shuffle(
    actual_true_labels, 
    predicted_clusters_kmeans, 
    predicted_clusters_spectral, 
    random_state=42
)

# Assuming `actual_true_labels` contains your true class labels
unique_categories = np.unique(actual_true_labels)

# Evaluate mappings for K-Means
best_mapping_kmeans, best_accuracy_kmeans = evaluate_mappings(
    actual_true_labels, 
    predicted_clusters_kmeans, 
    unique_categories,  # Make sure to pass the list of unique categories
    num_samples=100000  # Optional: adjust as needed or remove if not using sampling
)

# Evaluate mappings for Spectral Clustering
best_mapping_spectral, best_accuracy_spectral = evaluate_mappings(
    actual_true_labels, 
    predicted_clusters_spectral, 
    unique_categories,  # Make sure to pass the list of unique categories
    num_samples=100000  # Optional: adjust as needed or remove if not using sampling
)

# Evaluate mappings for Agglomerative Clustering
best_mapping_agglo, best_accuracy_agglo = evaluate_mappings(
    actual_true_labels, 
    final_clusters_agglo, 
    unique_categories,  # Make sure to pass the list of unique categories
    num_samples=100000  # Optional: adjust as needed or remove if not using sampling
)
# Print the results
print(f'Best Mapping for K-Means: {best_mapping_kmeans}')
print(f'Best Accuracy for K-Means: {best_accuracy_kmeans:.2%}')
print(f'Best Mapping for Spectral Clustering: {best_mapping_spectral}')
print(f'Best Accuracy for Spectral Clustering: {best_accuracy_spectral:.2%}')
print(f'Best Mapping for Agglomerative Clustering: {best_mapping_agglo}')
print(f'Best Accuracy for Agglomerative Clustering: {best_accuracy_agglo:.2%}')
