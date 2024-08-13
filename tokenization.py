# Tokenization or vectorization

from sklearn.feature_extraction.text import CountVectorizer

sample_data = ['this is the first paper.','This paper is the second paper','And this is the third one.','Is this the first paper?']
sample_other_data = ['Hello World', 'Hello Hello World', 'Hello World world world']
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(sample_other_data)
print(vectorizer.get_feature_names_out())
print(x.toarray())