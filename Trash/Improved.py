# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 08:37:49 2023

@author: olive
"""

#Â Libraries 
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
#from yellowbrick.cluster import KElbowVisualizer

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
#Â Dataset is too big, 
shuffled_df = df.sample(frac=0.25, random_state=42)
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
from nltk.corpus.reader.wordnet import NOUN
import nltk

# Define a stemmer for stemming words.
stemmer = PorterStemmer()

# Define a lemmatizer for lemmatizing words.
lemmatizer = WordNetLemmatizer()

# Define regex pattern for matching emojis.
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+",
    flags=re.UNICODE)

# Placeholder function for language detection (to be replaced with langdetect or similar in a different environment)
from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'  # In case language detection fails

# Function to map NLTK's part of speech tags to WordNet's part of speech tags.
# This is used in lemmatization to obtain the lemma of a word based on its part of speech.
def get_wordnet_pos(tag):
    """
    Convert the part-of-speech naming scheme from the Penn Treebank tags to WordNet's format.
    
    :param tag: The part-of-speech tag obtained from NLTK's part-of-speech tagger.
    :return: A character that represents the corresponding WordNet part of speech.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return NOUN

# Define a function to preprocess text by tokenizing, removing stopwords, stemming, and lemmatizing.
def preprocess_text(text, remove_stopwords=True, use_stemming=False, use_lemmatization=False):
    """
    Preprocess a given text using tokenization, stopwords removal, stemming, and lemmatization.
    
    :param text: The text to preprocess.
    :param remove_stopwords: If True, stopwords are removed from the text.
    :param use_stemming: If True, words are reduced to their stem form.
    :param use_lemmatization: If True, words are reduced to their base or dictionary form.
    :return: The preprocessed text as a string.
    """
    
    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    
    # Tokenize text using NLTK's word_tokenize function.
    tokens = word_tokenize(text)

    # Remove stopwords if the corresponding option is enabled.
    if remove_stopwords:
        tokens = [w for w in tokens if not w.lower() in stopwords.words('english')]

    # Apply stemming if the corresponding option is enabled.
    if use_stemming:
        tokens = [stemmer.stem(token) for token in tokens]

    # Apply lemmatization if the corresponding option is enabled.
    if use_lemmatization:
        pos_tags = nltk.pos_tag(tokens)
        tokens = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(pos)) for token, pos in pos_tags]

    # Rejoin tokens into a single string.
    text = ' '.join(tokens)
    return text

# Define a function to clean tweets by removing URLs, mentions, hashtags, and non-ASCII characters.
def clean_tweet(tweet, remove_stopwords=True, use_stemming=False, use_lemmatization=True):
    """
    Clean the tweet by removing URLs, mentions, hashtags, non-ASCII characters,
    and applying text preprocessing (tokenization, stopwords removal, stemming, lemmatization).
    
    :param tweet: The original tweet text.
    :param remove_stopwords: If True, stopwords are removed from the text.
    :param use_stemming: If True, words are reduced to their stem form.
    :param use_lemmatization: If True, words are reduced to their base or dictionary form.
    :return: The cleaned tweet text.
    """
    # Remove URLs, mentions, and hashtags from the tweet.
    tweet = re.sub(r'(#\w+)|(@\w+)|(\w+:\/\/\S+)', '', tweet)
    # Remove non-ASCII characters from the tweet.
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
    # Preprocess the tweet text.
    tweet = preprocess_text(tweet, remove_stopwords, use_stemming, use_lemmatization)
    return tweet

# Define a function that combines cleaning tweets with extracting features such as hashtags and mentions.
# Define a function that combines cleaning tweets with extracting features such as hashtags, mentions, emojis, and language.
def preprocess_tweet(tweet):
    """
    Preprocess the tweet by extracting certain features and then cleaning the text.
    
    :param tweet: The original tweet text.
    :return: A tuple containing the cleaned tweet text and a dictionary of extracted features.
    """
    # Extract features from the tweet.
    features = {
        'hashtags': re.findall(r'#\w+', tweet),  # Find all hashtags.
        'mentions': re.findall(r'@\w+', tweet),  # Find all mentions.
        'emojis': EMOJI_PATTERN.findall(tweet),  # Extract emojis using the defined regex pattern.
        'language': detect_language(tweet)       # Detect language using the placeholder function.
    }

    # Clean the tweet using the clean_tweet function.
    cleaned_tweet = clean_tweet(tweet)
    
    # Return both the cleaned tweet and the extracted features.
    return cleaned_tweet, features

from tqdm import tqdm
tqdm.pandas()
# Apply preprocessing to the tweets and time the process.
t1 = time.time()
print('Starting cleaning of data')

# Use tqdm to show progress bar as we apply preprocess_tweet function to each tweet in the DataFrame.
slice_df['cleaned'], slice_df['features'] = zip(*slice_df['tweet_text'].progress_apply(preprocess_tweet))
    
print('Finished cleaning of data')
t2 = time.time()
print(f'Elapsed time for preprocessing: {t2-t1:.2f}s')

# Extract the features from the features column into separate columns in the DataFrame.
slice_df['hashtags'] = slice_df['features'].apply(lambda x: x['hashtags'])
slice_df['mentions'] = slice_df['features'].apply(lambda x: x['mentions'])
# Add code to extract emojis and language if you have that functionality.
slice_df['emojis'] = slice_df['features'].apply(lambda x: x['emojis'])
slice_df['language'] = slice_df['features'].apply(lambda x: x['language'])

# Drop the 'features' column from the DataFrame as it's no longer needed.
slice_df.drop(columns=['features'], inplace=True)
# %%
# First, we define a function that combines all the features into a single string.
def combine_features(row):
    # Combine the cleaned text with hashtags, mentions, and emojis separated by spaces.
    # The space (' ') separator is important to ensure each token is treated separately in the vectorization.
    return ' '.join(row['hashtags']) + ' ' + ' '.join(row['mentions']) + ' ' + ' '.join(row['emojis']) + ' ' + row['cleaned']

# Now, apply the combine_features function to each row in the DataFrame to create a new 'combined_features' column.
# We use axis=1 to indicate that the function should be applied to each row.
slice_df['combined_features'] = slice_df.apply(combine_features, axis=1)

# We'll reinitialize the TfidfVectorizer with the same parameters as before.
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=0.01, max_df=0.85)

# We fit_transform the vectorizer to the 'combined_features' column to create the feature matrix.
X_text = vectorizer.fit_transform(slice_df['combined_features'])

# We can check the shape of the new feature matrix to understand the size of our dataset now.
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
#Set number of clusters for Kmeans and spectral
n = 5
#M## Method 1: K-means (own and sklearn kmeans)
###Â Using own K-means model
kmeans = KMeans(n_clusters=n, max_iter=300, random_state=42)
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
kmeans = KMeans(n_clusters=n, max_iter=300, random_state=42)
kmeans.fit(X_text.toarray())
slice_df['cluster_kmeans'] = kmeans.labels_

# %% [markdown]
# ### Method 2: Spectral clustering
from sklearn.cluster import SpectralClustering

gamma_value = 1  # Example gamma value, this needs to be tuned

# Initialize SpectralClustering with the RBF kernel and gamma parameter
spectral = SpectralClustering(
    n_clusters=n,
    random_state=42,
    affinity='rbf',
    gamma=gamma_value
)

# Fit the model to the TF-IDF features
clusters_spectral = spectral.fit_predict(X_text)

# Add the cluster labels to your DataFrame
slice_df['cluster_spectral'] = clusters_spectral




# %%
# ### Method 3: Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD

# Convert to a dense array if your dataset is small (DANGEROUS!!!! TAKES LONG TIME)
#X_dense = X_text.toarray() 

from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Initialize a range for the number of components we want to test
n_components_range = range(5, 355, 5)

# Initialize a list to store the explained variance for each number of components
explained_variances = []

# For each number of components, create a TruncatedSVD instance and fit_transform it to the TF-IDF matrix
# Then, append the total explained variance to the list
for n_components in n_components_range:
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(X_text)
    explained_variances.append(svd.explained_variance_ratio_.sum())

# Plotting the explained variances
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, explained_variances, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Total Explained Variance')
plt.title('Explained Variance by Different Number of Components')
plt.grid(True)
plt.show()
#%%
# Reduce the dimensionality of the data
svd = TruncatedSVD(n_components=120)  # Choose the number of components such as 100
X_reduced = svd.fit_transform(X_text)

# Assuming X_text is your TF-IDF or count vectorized data
linkage = 'average'
agglomerative = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage=linkage)
clusters_agglo = agglomerative.fit_predict(X_reduced)

# Evaluate the optimal number of clusters by silhouette score
range_n_clusters = list(range(3, 10))  # Example range from 2 to 9
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

#print('Cluster mappings for kmeans:')
#print(cluster_map_kmeans)

#print('Cluster mappings for spectral:')
#print(cluster_map_spectral)

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

# %%
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
from sklearn.metrics import silhouette_score

# Silhouette Score for own K-means model
silhouette_kmeans_own = silhouette_score(X_text, clusters_kmeans)
print(f"Silhouette Score for own K-Means: {silhouette_kmeans_own}")

# Silhouette Score for Sklearn K-means
silhouette_kmeans = silhouette_score(X_text, slice_df['cluster_kmeans'])
print(f"Silhouette Score for sklearn K-Means: {silhouette_kmeans}")

# Silhouette Score for Spectral Clustering
silhouette_spectral = silhouette_score(X_text, clusters_spectral)
print(f"Silhouette Score for Spectral Clustering: {silhouette_spectral}")

# Silhouette Score for Agglomerative Clustering with the optimal number of clusters
silhouette_agglo = silhouette_score(X_reduced, final_clusters_agglo)
print(f"Silhouette Score for Agglomerative Clustering with {best_n_clusters} clusters: {silhouette_agglo}")

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
cluster_map_agglo_auto = cluster_map_agglo

# Assuming you have t-SNE features stored in 'tsne-2d-one' and 'tsne-2d-two' columns.
# If not, you need to create these columns from your t-SNE embedding:
slice_df['tsne-2d-one'] = X_embedded[:, 0]
slice_df['tsne-2d-two'] = X_embedded[:, 1]

# Define the color palette with 5 distinct colors
distinct_colors = sns.color_palette('tab10', n_colors=5)

# Now you can use this palette in your scatterplot function
def plot_tnse_clusters(slice_df, method_name, cluster_mapping, X_embedded):
    plt.figure(figsize=(12, 8))
    plt.title(f"t-SNE + {method_name} Cyberbullying Tweets Clustering", fontdict={"fontsize": 18})
    plt.xlabel("X0", fontdict={"fontsize": 16})
    plt.ylabel("X1", fontdict={"fontsize": 16})

    # Create the scatter plot using the distinct color palette
    sns.scatterplot(
        x=X_embedded[:, 0] + np.random.uniform(-0.5, 0.5, X_embedded.shape[0]),  # Jitter X
        y=X_embedded[:, 1] + np.random.uniform(-0.5, 0.5, X_embedded.shape[0]),  # Jitter Y
        hue=slice_df[cluster_mapping],
        palette=distinct_colors,  # Using the distinct color palette
        alpha=0.6,
        edgecolor='w',
        linewidth=0.5,
        s=50
    )

    # Adjust the legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title='Bullying Classes')
    plt.show()

# Apply the mappings to assign the predicted cyberbullying types based on clusters.
slice_df['cluster_mapped_kmeans'] = slice_df['cluster_kmeans'].map(cluster_map_kmeans_auto)
slice_df['cluster_mapped_spectral'] = slice_df['cluster_spectral'].map(cluster_map_spectral_auto)
slice_df['cluster_mapped_agglo'] = slice_df['cluster_agglo'].map(cluster_map_agglo_auto)

# Create the t-SNE scatter plots for each clustering method
plot_tnse_clusters(slice_df, 'K-Means', 'cluster_mapped_kmeans', X_embedded)
plot_tnse_clusters(slice_df, 'Spectral', 'cluster_mapped_spectral', X_embedded)
plot_tnse_clusters(slice_df, 'Agglomerative', 'cluster_mapped_agglo', X_embedded)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

def plot_clusters_with_misclassification(slice_df, true_label_column, predicted_label_column, method_name):
    # Identify misclassified rows
    slice_df[f'misclassified_{method_name}'] = slice_df[true_label_column] != slice_df[predicted_label_column]
    
    # Create a scatter plot for correctly classified points
    plt.figure(figsize=(16, 10))
    # Plot correctly classified points
    correctly_classified = sns.scatterplot(
        x='tsne-2d-one',
        y='tsne-2d-two',
        data=slice_df[~slice_df[f'misclassified_{method_name}']],
        hue=predicted_label_column,
        palette=distinct_colors,
        alpha=0.7,
        legend='full'
    )

    # Plot misclassified points with original cluster color
    misclassified_original = sns.scatterplot(
        x='tsne-2d-one',
        y='tsne-2d-two',
        data=slice_df[slice_df[f'misclassified_{method_name}']],
        hue=predicted_label_column,
        palette=distinct_colors,
        alpha=0.7,
        legend=False,
        edgecolor=None,  # No edge color for this layer
        s=50  # Slightly larger to stand out with red border
    )

    # Overlay misclassified points with red borders
    misclassified_red_border = sns.scatterplot(
        x='tsne-2d-one',
        y='tsne-2d-two',
        data=slice_df[slice_df[f'misclassified_{method_name}']],
        color='none',  # No fill color
        edgecolor='red',
        s=60,  # Slightly larger to create a visible border
        legend=False
    )

    # Add titles and labels
    plt.title(f'{method_name} Clustering Visualization with t-SNE')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')

    # Create custom legend handles
    handles, labels = correctly_classified.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker='o', color='w', label='Misclassified',
                          markerfacecolor='none', markeredgecolor='red', markersize=10))

    # Create and adjust the legend
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()

# Now let's plot for each clustering method
plot_clusters_with_misclassification(slice_df, 'cyberbullying_type', 'cluster_mapped_kmeans', 'K-Means')
plot_clusters_with_misclassification(slice_df, 'cyberbullying_type', 'cluster_mapped_spectral', 'Spectral')
plot_clusters_with_misclassification(slice_df, 'cyberbullying_type', 'cluster_mapped_agglo', 'Agglomerative')







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
# ## Analysis of misclasified points
# %%
#Find the list of misclassified indexes 
diff_list_kmeans = np.where(slice_df['cyberbullying_type']!= slice_df['cluster_mapped_kmeans'])
misclassified_kmeans = slice_df[slice_df.index.isin(diff_list_kmeans[0])]
print(len(diff_list_kmeans[0]))



######Â Figure for visuaization of misclasification #########

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
