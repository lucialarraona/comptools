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



############# First approach we used ##############
def preprocess_tweet(text, remove_stopwords=True):
    """Preprocess tweet text data by:
        - removing URLs
        - removing mentions and hashtags
        - removing special characters
        - removing numbers
        - removing stopwords (optional)
        - transforming to lowercase
        - removing excessive whitespaces
        - replacing line separators with spaces
    Args:
        text (str): The input tweet text to clean.
        remove_stopwords (bool): Whether or not to remove stopwords (default: True).
    Returns:
        str: The cleaned tweet text.
    """
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove mentions and hashtags (keep the words)
    text = re.sub(r'[@#]\w+', '', text)
    
    # Remove special characters and numbers
    text = re.sub('[^A-Za-z]+', ' ', text)
    
    # Replace line separators with spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    if remove_stopwords:
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [w for w in tokens if w.lower() not in set(stopwords.words("english"))]
        # Join back together
        text = " ".join(tokens)
    
    # Transform to lowercase
    text = text.lower()
    
    # Remove excessive whitespaces
    text = ' '.join(text.split())
    
    return text



#Apply this function to the tweets (also measure time it takes for computational metric purposes)
t1 = time.time()
print('Starting cleaning of data')
tqdm.pandas(dynamic_ncols=True, smoothing=0.01)
#Use function preprocess_text() for every row, assign results to new column 
slice_df['cleaned'] = slice_df['tweet_text'].progress_apply(lambda x: preprocess_tweet(x, remove_stopwords=True))
print('Finished cleaning of data')
t2 = time.time()
print(f'Elapsed time for initilization: {t2-t1:.2f}s')






####### SVM
# Now we'll vectorize the 'tweet_text' column of slice_df using the previously mentioned TF-IDF parameters
# Since we don't have the actual 'cleaned' column, we use 'tweet_text' for the purpose of this demonstration
X_text_slice = X_text

label_encoder = LabelEncoder()
# Encode the labels for the sliced dataset
y_labels_slice = label_encoder.fit_transform(slice_df['cyberbullying_type'])

# Check the shapes of the features and labels
X_text_slice.shape, y_labels_slice.shape

# Step 1: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_text_slice, y_labels_slice, test_size=0.2, random_state=42)

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