# Libraries 
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import warnings
import networkx as nx
import re
import time
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import random 
from yellowbrick.cluster import KElbowVisualizer

warnings.filterwarnings("ignore")

# %% [markdown]
# # Exploratory Data Analysis (EDA) 
# - Data Import / Cleaning and Vectorization
# - Distribution of datapoints 

# %%
# Spotify tracks dataset (for recommendation system)
df = pd.read_csv("C:/Users/olive/OneDrive - Danmarks Tekniske Universitet/Masters - Courses/02807 - Computational Tools for Data Science/comptools/Data/cyberbullying_tweets 2.csv", index_col=0)
df = df.dropna()
df = df.reset_index()
df.head(2)

# %%
# Data Analysis
print(df.shape)
df['cyberbullying_type'].unique()

# %%
df['cyberbullying_type'].head

# %%
# Define the category you want to drop
category_to_drop = 'other_cyberbullying'

# Filter and create a new DataFrame without the specified category
df = df[df['cyberbullying_type'] != category_to_drop]
df.duplicated().sum()
df = df[~df.duplicated()]

# %%
# Dataset is too big, 
shuffled_df = df.sample(frac=1, random_state=42)
slice_df = shuffled_df
slice_df = slice_df.reset_index()

# %%
#Plot categories distribution
slice_df['cyberbullying_type'].value_counts().plot.bar(rot=0, color='pink')
plt.title('Categorie distribution of cyberbullying')
plt.ylabel('# of tweets')
None

# %%
slice_df.head

# %%
# Text Cleaning 
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Helper function to convert NLTK's part of speech tags to wordnet's part of speech tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # By default, NLTK's POS tagger assigns NN tag
'''

Two different version, gotta test which is best

'''
def preprocess_text(text, remove_stopwords):
    """This utility function sanitizes a string by:
        - removing links
        - removing special characters
        - removing numbers
        - removing stopwords
        - transforming in lowercase
        - removing excessive whitespaces
        Args:
            text (str): the input text you want to clean
            remove_stopwords (bool): whether or not to remove stopwords
        Returns:
            str: the cleaned text
        """
    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    if remove_stopwords:
        # 1. tokenize
        tokens = word_tokenize(text)
        # 2. check if stopword
        tokens = [
            w for w in tokens if not w.lower() in stopwords.words("english")
        ]
        # 3. join back together
        text = " ".join(tokens)
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
    return text

def preprocess_text(text, remove_stopwords=True, use_stemming=False, use_lemmatization=False):
    # Your existing preprocessing steps here

    # Tokenize text
    tokens = word_tokenize(text)

    if remove_stopwords:
        # Remove stopwords
        tokens = [w for w in tokens if not w.lower() in stopwords.words('english')]

    if use_stemming:
        # Stem tokens
        tokens = [stemmer.stem(token) for token in tokens]

    if use_lemmatization:
        # Lemmatize tokens
        pos_tags = nltk.pos_tag(tokens)
        tokens = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(pos)) for token, pos in pos_tags]

    # Rejoin tokens
    text = ' '.join(tokens)
    return text


#Apply this function to the song lyrics ()
from tqdm import tqdm
tqdm.pandas()

t1 = time.time()
print('Starting cleaning of data')

# Update the following line to include the additional arguments for stemming and lemmatization
slice_df['cleaned'] = slice_df['tweet_text'].progress_apply(lambda x: preprocess_text(x, remove_stopwords=True, use_stemming=True, use_lemmatization=True))

print('Finished cleaning of data')
t2 = time.time()
print(f'Elapsed time for preprocessing: {t2-t1:.2f}s')

# %%
# Text vectorization using sklearn tfidf vectorizer to the preprocessed cleaned data
#vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=0.01, max_df=0.85)

# fit_transform applies TF-IDF to cleaned texts 
X_text = vectorizer.fit_transform(slice_df['cleaned'])
X_text.shape

# %%
# Reduce the dimensionality of data points to plot the datapoint distribution
pca = PCA(n_components=50, random_state=42)
# pass our X to the pca and store the reduced vectors into pca_vecs
pca_vecs = pca.fit_transform(X_text.toarray())
X_embedded = TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(pca_vecs)
                  

plt.title("T-SNE class representation", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=slice_df,
                x=X_embedded[:, 0],
                y=X_embedded[:, 1],
                hue='cyberbullying_type',
                palette="cubehelix")
#plt.legend(loc='upper right')
plt.legend(bbox_to_anchor=(1.05, 1),
           loc='upper left',
           borderaxespad=1,
           title='Type of cyberbullying')
plt.show()

# %%
def get_top_keywords(X, clusters, vectorizer, n_terms):
    """This function returns the keywords for each centroid of the KMeans"""
    dff = pd.DataFrame(X.todense()).groupby(clusters).mean()  # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names_out()  # access tf-idf terms
    for i, r in dff.iterrows():
        print('\nCluster {}'.format(i))
         # for each row of the dataframe, find the n terms that have the highest tf idf score
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) 

# %%
print('The most central words for each category')
get_top_keywords(X_text, slice_df['cyberbullying_type'], vectorizer, 10)

# %% [markdown]
# # Clustering

# %% [markdown]
# ### Method  1: K-Means

# %%
class KMeans:

    def __init__(self, n_clusters, max_iter=300, random_state=1312):
        """
        Parameters
        ----------
        n_clusters : INT
            Number of clusters for K-means
        max_iter : INT, optional
            Number of iterations run by K-means. The default is 300.
        random_state : INT, optional
            Random state for initilization. Used for replication.

        Returns
        -------
        None.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initCentroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def getCentroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def getDist(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = np.linalg.norm(X - centroids[k, :], axis=1) #default is frobenius norm, which is equivilant to 2-norm for vectors
            distance[:, k] = np.square(row_norm)
        return distance
    
    def fit(self, X):
        self.centroids = self.initCentroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.getDist(X, old_centroids)
            self.labels = np.argmin(distance, axis=1)
            self.centroids = self.getCentroids(X, self.labels)
            if np.all(old_centroids == self.centroids): #If no updates are done
                break

    def predict(self, X):
        distance = self.getDist(X, self.centroids)
        return np.argmin(distance,axis=1)

# %%
#M## Method 1: K-means (own and sklearn kmeans)
### Using own K-means model
kmeans = KMeans(n_clusters=5, max_iter=300, random_state=42)
# fit the model
kmeans.fit(X_text.toarray())
# store cluster labels in a variable
clusters_kmeans = kmeans.predict(X_text.toarray())

print(clusters_kmeans.shape)
#add a column with clusters assigned by kmeans
slice_df['cluster_kmeans_own'] = clusters_kmeans

# %%
#### Using Sklearn kmeans 
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_text.toarray())
slice_df['cluster_kmeans'] = kmeans.labels_


# %% [markdown]
# ### Method 2: Spectral clustering
from sklearn.cluster import SpectralClustering
# Initialize SpectralClustering
n_clusters = 5
spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')

# Fit the model to the TF-IDF features
spectral_cluster_labels = spectral.fit(X_text)

# Retrieve the cluster labels
clusters_spectral = spectral.labels_

# Add the cluster labels to your DataFrame
slice_df['cluster_spectral'] = clusters_spectral

# %%
# ### Method 3: Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD

# Convert to a dense array if your dataset is small (DANGEROUS!!!! TAKES LONG TIME)
#X_dense = X_text.toarray() 

# Reduce the dimensionality of the data
svd = TruncatedSVD(n_components=100)  # Choose the number of components such as 100
X_reduced = svd.fit_transform(X_text)

# Assuming X_text is your TF-IDF or count vectorized data
agglomerative = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage='ward')
clusters_agglo = agglomerative.fit_predict(X_reduced)

# Evaluate the optimal number of clusters by silhouette score
range_n_clusters = list(range(2, 10))  # Example range from 2 to 9
best_score = -1
best_n_clusters = 0

for n_clusters in range_n_clusters:
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    preds = clusterer.fit_predict(X_reduced)
    score = silhouette_score(X_reduced, preds)

    if score > best_score:
        best_score = score
        best_n_clusters = n_clusters

# Final model with the optimal number of clusters
final_agglomerative = AgglomerativeClustering(n_clusters=best_n_clusters)
final_clusters_agglo = final_agglomerative.fit_predict(X_reduced)

slice_df['cluster_agglo'] = final_clusters_agglo

# %%
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

vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
X_text = vectorizer.fit_transform(slice_df['cleaned'])

# Example usage:
top_words_category = calculate_top_words_per_category(slice_df, slice_df['cyberbullying_type'].unique(), vectorizer, 20)
top_words_cluster_kmeans = calculate_top_words_per_cluster(X_text, slice_df['cluster_kmeans'], vectorizer, 20)
top_words_cluster_spectral = calculate_top_words_per_cluster(X_text, slice_df['cluster_spectral'], vectorizer, 20)
top_words_cluster_agglo = calculate_top_words_per_cluster(X_text, final_clusters_agglo, vectorizer, 20)

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

'''

Finding cluster mappings using top N words from correct labeled data and then assign a bullying type to each cluster
based on top N words from each cluster.

'''
# Assuming you have `top_words_cluster_kmeans` and `top_words_cluster_spectral`
# from your previous operations:
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

print('Cluster mappings for Agglo:')
print(cluster_map_agglo)

# Apply the mappings
slice_df['cluster_mapped_kmeans'] = slice_df['cluster_kmeans'].map(cluster_map_kmeans)
slice_df['cluster_mapped_spectral'] = slice_df['cluster_spectral'].map(cluster_map_spectral)
slice_df['cluster_mapped_agglo'] = slice_df['cluster_agglo'].map(cluster_map_agglo)

# Find the correctly classified rows for K-Means
correct_list_kmeans = slice_df[slice_df['cyberbullying_type'] == slice_df['cluster_mapped_kmeans']].index.tolist()
# Filter to only include correctly classified rows for K-Means
correctly_classified_kmeans = slice_df.loc[correct_list_kmeans]

# Find the correctly classified rows for Spectral Clustering
correct_list_spectral = slice_df[slice_df['cyberbullying_type'] == slice_df['cluster_mapped_spectral']].index.tolist()
# Filter to only include correctly classified rows for Spectral Clustering
correctly_classified_spectral = slice_df.loc[correct_list_spectral]

# Find the correctly classified rows for Agglomerative Clustering
correct_list_agglo = slice_df[slice_df['cyberbullying_type'] == slice_df['cluster_mapped_agglo']].index.tolist()
# Filter to only include correctly classified rows for Agglomerative Clustering
correctly_classified_agglo = slice_df.loc[correct_list_agglo]

# Calculate the accuracy for both methods
accuracy_kmeans = len(correctly_classified_kmeans) / len(slice_df)
accuracy_spectral = len(correctly_classified_spectral) / len(slice_df)
accuracy_agglo = len(correctly_classified_agglo) / len(slice_df)
print(f'Accuracy of K-Means clustering: {accuracy_kmeans:.2%}')
print(f'Accuracy of Spectral clustering: {accuracy_spectral:.2%}')
print(f'Accuracy of Agglomerative Clustering: {accuracy_agglo:.2%}')

'''

Testing every single combination of mappings

'''
from itertools import permutations, product

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

from sklearn.utils import shuffle
import random

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


# %%
'''
Use manual if you want to look at top 10 words yourself and categorize
'''
cluster_map_kmeans_manual = {
    0: "age",
    1: "not_cyberbullying",
    2: "ethnicity",
    3: "religion",
    4: "gender"
}

cluster_map_spectral_manual = {
    0: "age",
    1: "not_cyberbullying",
    2: "ethnicity",
    3: "religion",
    4: "gender"
}
'''
Use auto variables if you want to use the cluster mappings based on the code above
'''

cluster_map_kmeans_auto = cluster_map_kmeans
cluster_map_spectral_auto = cluster_map_spectral

# apply mapping
slice_df['cluster_mapped_kmeans'] = slice_df['cluster_kmeans'].map(cluster_map_kmeans)

plt.title("TF-IDF + K-Means Cyberbullying tweets clustering", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=slice_df,
                x=X_embedded[:, 0],
                y=X_embedded[:, 1],
                hue='cluster_mapped_kmeans',
                palette="cubehelix")
#plt.legend(loc='upper right')
plt.legend(bbox_to_anchor=(1.05, 1),
           loc='upper left',
           borderaxespad=1,
           title='Bullying classes')
plt.show()











# %%
from nltk.sentiment import SentimentIntensityAnalyzer

# Assuming 'correctly_classified_kmeans' DataFrame is defined from previous steps
# and 'cluster_map_kmeans' is a dictionary with cluster ids mapped to category names.

# Initialize VADER's SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Function to get the VADER sentiment score
def get_vader_sentiment(tweet):
    return sid.polarity_scores(tweet)['compound']

# Apply VADER sentiment analysis to each correctly classified tweet
correctly_classified_kmeans['vader_sentiment_score'] = correctly_classified_kmeans['cleaned'].apply(get_vader_sentiment)

# Find the most negative tweet in each cluster among the correctly classified
most_negative_tweets_vader = {}

for cluster_label, cluster_name in cluster_map_kmeans.items():
    cluster_data = correctly_classified_kmeans[correctly_classified_kmeans['cluster_mapped_kmeans'] == cluster_name]
    if not cluster_data.empty:
        most_negative_tweet = cluster_data.loc[cluster_data['vader_sentiment_score'].idxmin()]
        most_negative_tweets_vader[cluster_name] = most_negative_tweet

# Create a list to display the results later
negative_tweet_info = []

# Display the most negative tweet for each cluster among the correctly classified
for cluster_name, tweet_data in most_negative_tweets_vader.items():
    negative_tweet_info.append({
        "Cluster": cluster_name,
        "Tweet": tweet_data['tweet_text'],
        "VADER Sentiment Score": tweet_data['vader_sentiment_score']
    })

negative_tweet_info


# %% [markdown]
'''
DBSCAN SUCKS!!!!
# ### Method 3: DBSCAN
from sklearn.cluster import DBSCAN

# Initialize DBSCAN with the chosen parameters
dbscan_chosen = DBSCAN(eps=0.9, min_samples=3)

# Fit the DBSCAN model to the vectorized text data
clusters_chosen = dbscan_chosen.fit_predict(X_text)

# The cluster labels for each point
cluster_labels_chosen = dbscan_chosen.labels_

# Count the number of points in each cluster
clusters_count = pd.Series(cluster_labels_chosen).value_counts().sort_index()

# For interpretation, let's extract some tweet texts from each cluster
# We will create a DataFrame that includes the cluster labels and the tweet texts
clustered_tweets = pd.DataFrame({'cluster': cluster_labels_chosen, 'cleaned': slice_df['cleaned']})

# Sample a few tweets from each cluster for interpretation
sample_tweets_per_cluster = clustered_tweets.groupby('cluster').head(3).sort_values(by='cluster').reset_index(drop=True)

clusters_count, sample_tweets_per_cluster
'''

# %% [markdown]
# ## Analysis of misclasified points

# %%
#Find the list of misclassified indexes 
diff_list_kmeans = np.where(slice_df['cyberbullying_type']!= slice_df['cluster_mapped_kmeans'])
misclassified_kmeans = slice_df[slice_df.index.isin(diff_list_kmeans[0])]
print(len(diff_list_kmeans[0]))



###### Figure for visuaization of misclasification #########

figs, ax = plt.subplots(2,layout="constrained")
#Distribution of misclassified points - true category
misclassified_kmeans['cyberbullying_type'].value_counts().plot(ax=ax[0],
                                   kind='bar',
                                   rot=0,
                                   color=["blue","deepskyblue","teal","turquoise"],
                                   edgecolor=["gray"],
                                   title = 'True category distribution of misclassified points' )


#Distribution of misclassified points - predicted cluster
misclassified_kmeans['cluster_mapped_kmeans'].value_counts().plot(ax=ax[1],
                                   kind='bar',
                                   rot=0,
                                   color=["blue","deepskyblue","teal","turquoise"],
                                   edgecolor=["gray"],
                                   title = 'Predicted cluster distribution of misclassified points' )

#plt.savefig('outlier_anal.png', dpi=300)
None

# %% [markdown]
# ## Clasification
# - Neural Network
# - Transformer (BERT)

# %% [markdown]
# ### Neural Network

# %% [markdown]
# 

# %%


# %% [markdown]
# ### BERT 

# %% [markdown]
# 

# %% [markdown]
# ### SVM

# %%
from sklearn.svm import SVC
# Encode the labels for the sliced dataset
y_labels_slice = label_encoder.fit_transform(slice_df['cyberbullying_type'])

# Check the shapes of the features and labels
X_text.shape, y_labels_slice.shape

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Step 1: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_text, y_labels_slice, test_size=0.2, random_state=42)

# Initialize the SVM classifier with a linear kernel
svm = SVC(kernel='linear', random_state=42)

# Train the SVM classifier on the training data
svm.fit(X_train, y_train)

# Predict on the test set
y_pred_svm = svm.predict(X_test)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)

accuracy_svm, conf_matrix_svm, report_svm



