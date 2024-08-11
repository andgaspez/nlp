import pandas as pd 
import numpy as np 
import seaborn as sns 
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

# Load the data
tweets_df = pd.read_csv('twitter.csv')

# Preparing the text data
sentences = tweets_df['tweet'].tolist()
sentences_as_huge_string = ''.join(sentences)

# World clouds instances
wordcloud_sentences = WordCloud(width=2000, height=1000).generate(sentences_as_huge_string)
wordcloud_positive = WordCloud(width=2000, height=1000).generate(''.join(tweets_df[tweets_df['label'] == 0]['tweet'].tolist()))
wordcloud_negative = WordCloud(width=2000, height=1000).generate(''.join(tweets_df[tweets_df['label'] == 1]['tweet'].tolist()))

# List of worldcloud instances
wordclouds = [
    (wordcloud_sentences, 'Sentiment Analysis: Sentences Word Cloud'),
    (wordcloud_positive, 'Sentiment Analysis: Positive Sentences Word Cloud'),
    (wordcloud_negative, 'Sentiment Analysis: Negative Sentences Word Cloud')
]

# Word clouds plots
for i, (wordcloud, title) in enumerate(wordclouds):
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=10)
    plt.show(block=False)  # Non-blocking display

plt.show()  # Ensure all figures are shown










