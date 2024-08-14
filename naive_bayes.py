import string, nltk # Natural Languaje Toolkit
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

def message_cleaning(message):
    
    message_punc_cleaned = [char for char in message if char not in string.punctuation]
    message_punc_cleaned = ''.join(message_punc_cleaned)

    message_stopw_cleaned = [word for word in message_punc_cleaned.split() if word.lower() not in stopwords.words('english')]

    return message_stopw_cleaned


# Load the data
tweets_df = pd.read_csv('twitter.csv')

# Lets drop a column. Arg axis=0 means row, axis=1 means a column
tweets_df = tweets_df.drop(['id'], axis=1)

print(f'Original tweet: '+tweets_df['tweet'][5]) # Show the original up version

tweets_df_clean = tweets_df['tweet'].apply(message_cleaning)

print(f'Cleaned up tweet: {tweets_df_clean}') # Show the cleaned up version

vectorizer = CountVectorizer(analyzer = message_cleaning, dtype = np.uint8, max_features=12000)
tweets_countvectorizer = vectorizer.fit_transform(tweets_df['tweet'])
print(vectorizer.get_feature_names_out())
print(tweets_countvectorizer.toarray())
print(tweets_countvectorizer.shape)

X = pd.DataFrame(tweets_countvectorizer.toarray())
#print(X.shape)

Y = tweets_df['label']
#print(Y.shape)

from sklearn.model_selection import train_test_split #standard process whenever we train an AI or ML model
#we train with 80% of the data, the performance is assest in an unbiased way using new data never seen before (20%)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, Y_train)

from sklearn.metrics import classification_report, confusion_matrix

# Predicting the Test set results
Y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(Y_test, Y_predict_test)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(Y_test, Y_predict_test))