import string, nltk # Natural Languaje Toolkit
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

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

vectorizer = CountVectorizer(analyzer = message_cleaning, dtype = np.uint8)
tweets_countvectorizer = vectorizer.fit_transform(tweets_df['tweet'])
print(vectorizer.get_feature_names_out())
print(tweets_countvectorizer.toarray())
print(tweets_countvectorizer.shape)

X = pd.DataFrame(tweets_countvectorizer.toarray())
print(X)

# dummy_text = 'The #development of the @pipeline would be needed to remove stopwords and punctuations!'
# print(f'Original text: {message_cleaning(dummy_text)}')
# print(f'{message_cleaning(dummy_text)}')