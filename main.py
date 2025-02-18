import pandas as pd
import re
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Load dataset without headers
df = pd.read_csv(
    r'/home/muhammad-ahmad-nadeem/Projects/sentiment_analysis/resources/twitter_training.csv', 
    header=None
)

# Assign column names
df.columns = ['sr', 'platform', 'sentiment', 'comment']
df.head(3)
#  sr      platform     sentiment               comment
# 0  2401  Borderlands  Positive  im getting on borderlands and i will murder yo...
# 1  2401  Borderlands  Positive  I am coming to the borders and I will kill you...
# 2  2401  Borderlands  Positive  im getting on borderlands and i will kill you ...


# df.info()
 #   Column     Non-Null Count  Dtype 
# ---  ------     --------------  ----- 
#  0   sr         74682 non-null  int64 
#  1   platform   74682 non-null  object
#  2   sentiment  74682 non-null  object
#  3   comment    73996 non-null  object

df.shape
# (74682, 4)

df.duplicated().sum()
# 2700
# There are 2700 duplicate rows in the dataset
df = df.drop_duplicates()
df = df.fillna('')

df['sentiment'].unique() #['Positive' 'Neutral' 'Negative' 'Irrelevant']
df['platform'].nunique() #32

# Drop the first column
df = df.drop(['sr'], axis=1)
# print(df.head(3))

# Convert All Values to Strings
df['comment'] = df['comment'].astype(str)

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

# Clean the 'comment' column
df['comment'] = df['comment'].apply(clean_text)

'''Visualization'''
sns.countplot(data=df, x='sentiment')
plt.title('Sentiment Distribution')
# plt.show()

platform_sentiment = df.groupby(['platform', 'sentiment']).size().unstack()
platform_sentiment.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Platform Sentiment Distribution')
plt.ylabel('Count')
# plt.show()
df.head(3)


'''now i am droping the platform to make a prediction of new comment beside of company'''
df = df.drop(['platform'], axis=1)
# print(df.head(3))
#   sentiment                   comment
# 0  Positive  im getting on borderlands and i will murder yo...
# 1  Positive  i am coming to the borders and i will kill you...
# 2  Positive  im getting on borderlands and i will kill you all

# Encoding sentiment labels
df.head(3)
#    sentiment  comment
# 0      19811    35192
# 1      19811    28572
# 2      19811    35191

order = {'Positive': 1, 'Neutral': 2, 'Negative': 3, 'Irrelevant': 4}
df['sentiment'] = df['sentiment'].map(order)

# Extract features (X) and target labels (y)
x = df['comment']
y = df['sentiment']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

# Vectorize text using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Train a KNN model
knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric='euclidean')
knn.fit(x_train_vec, y_train)

'''hyperparameter tuning'''
# param_grid = {
#     'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 20],  # Number of neighbors
#     'weights': ['uniform', 'distance'],  # Weights for neighbors
#     'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metrics
# }

# grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# Fit GridSearchCV on the training data
# grid_search.fit(x_train, y_train)

# Get the best parameters from GridSearchCV
# best_params = grid_search.best_params_

# Print the best parameters
# print("Best parameters found by GridSearchCV:", best_params)

# Use the best model found by GridSearchCV to make predictions
# best_knn = grid_search.best_estimator_
# print(best_knn)
# print('*'*2)
# y_pred = best_knn.predict(x_test)

# Evaluate the model performance
# accuracy = grid_search.score(y_test, y_pred)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")

'''algo'''
# for i in range (1,30):
#     knn = KNeighborsClassifier(n_neighbors = i)
#     knn.fit(x_train,y_train)
#     print(i,knn.score(x_test, y_test),i,knn.score(x_train,y_train))


# Prediction function
def prediction(text):
    # Clean the text
    r = clean_text(text)
    # Transform the text using the vectorizer
    w = vectorizer.transform([r])  # Note: Transform, not fit_transform
    # Make the prediction
    w1 = knn.predict(w)
    
    if w1 == 1:
        return 'positive'
    elif w1 == 2:
        return 'neutral'
    elif w1 == 3:
        return "negative"
    else:
        return "aggressive"

# Test prediction
print(prediction("I absolutely love this product! It works like a charm and exceeded my expectations. Highly recommend it to everyone!"))
print(prediction("The product works fine, but it's nothing special. It does the job, but I wouldn't go out of my way to buy it again."))
print(prediction("This product is terrible. It broke after just one use, and customer service hasn't been helpful at all. I'm extremely disappointed."))
print(prediction("ust finished watching the latest episode of my favorite show. Can’t wait for the next season!"))

#positive
#neutral
#negative
#aggresive