import string
"""
Script to test the data cleaning and punctuation from a given text
"""
print(string.punctuation)
# Perform punctuation cleaning
Test = 'I love #somuch AI & @machinelearning!!!'
#print(Test)

test_punc_removed = [char for char in Test if char not in string.punctuation]
test_punc_removed_join = ''.join(test_punc_removed)
#print(test_punc_removed_join)

# Perform stop words cleaning
import nltk # Natural Languaje Toolkit
nltk.download('stopwords')

# It is needed to download stopwords package to execuse
from nltk.corpus import stopwords

#print(stopwords.words('english')) # stopwords.words('spanish') for ES languaje

Test_stopwords = 'I enjoy, coding, programming and Artificial intelligence'

Test_stopwords_clean = [word for word in Test_stopwords.split() if word.lower() not in stopwords.words('english')]

print(Test_stopwords_clean)