# 3. Sample Dataset
sample_text = "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate text in a way that is both meaningful and contextually appropriate. NLP techniques are widely used in applications such as chatbots, speech recognition, sentiment analysis, and machine translation. The process involves various steps, including tokenization, stopword removal, stemming, and lemmatization, which help break down and analyze text more effectively. Advances in deep learning and neural networks have significantly improved NLP models, making them more accurate and capable of handling complex linguistic structures. From customer support automation to content recommendation systems, NLP plays a crucial role in enhancing human-computer interaction and streamlining various business processes. As technology evolves, the potential of NLP continues to expand, driving innovation in fields like healthcare, finance, and education."

# Importing Libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Tokenization
words = word_tokenize(sample_text)
sentences = sent_tokenize(sample_text)

# Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

# Output Results
print("Original Text:", sample_text)
print("\nTokenized Words:", words)
print("\nTokenized Sentences:", sentences)
print("\nFiltered Words (Stopword Removal):", filtered_words)
print("\nStemmed Words:", stemmed_words)
print("\nLemmatized Words:", lemmatized_words)