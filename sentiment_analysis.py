"""
Goal: Build, train, test and deploy an AI model to predict sentiment from real 
Amazon Echo customer reviews.

Tool: Anaconda, Python, Scikit-Learn, Matplotlib, Searborn

Practical real world application:
AI/ML-based sentiment analysis is crucial for companies to automatically predict 
whether their customers are happy or not. The process could be done automatically
without having humans manually review thousands of customer reviews.

Data:
    Inputs: AMazon reviews
    Output: Sentiment (0 or 1)
"""

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from jupyterthemes import jtplot
#jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

# Set option to display all columns
pd.set_option('display.max_columns', None)

# If you also want to display all rows
pd.set_option('display.max_rows', None)

# Load the data
reviews_df = pd.read_csv('amazon_reviews.csv')

# View the dataframe information
print(reviews_df.info())

# View dataframe statistical summary
print(reviews_df.describe())

# Plot the count plot for the ratings
sns.countplot(x=reviews_df['rating'])
plt.show()

# Length of the verified_reviews column
length = len(reviews_df['verified_reviews'])
#reviews_df['length'] = reviews_df['verified_reviews'].apply(len)
print(length)
 