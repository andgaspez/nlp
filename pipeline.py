
import string, nltk # Natural Languaje Toolkit
nltk.download('stopwords')
from nltk.corpus import stopwords


dummy_text = 'The #development of the pipeline would be needed to remove stopwords and punctuations!'

dummy_text_cleaned = [char for char in dummy_text if char not in string.punctuation]
dummy_text_cleaned = ''.join(dummy_text_cleaned)

print(dummy_text_cleaned)

dummy_text_cleaned = [word for word in dummy_text_cleaned.split() if word.lower() not in stopwords.words('english')]
print(dummy_text_cleaned)
