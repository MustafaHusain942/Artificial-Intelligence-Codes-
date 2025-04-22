# Sample Dataset
sample_text = "Natural Language Processing (NLP) enables computers to understand human language. It is a key aspect of AI."

# Code, Input, Output
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required datasets
nltk.download('punkt')
nltk.download ('stopwords') 
nltk. download ('wordnet')

# Tokenization
words = word_tokenize(sample_text)
sentences = sent_tokenize(sample_text)

# Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word. lower() not in stop_words]

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
# Output Results
print("Origínal Text:", sample_text) 
print("\nInTokenized Words:", words)
print("\nTokenized Sentences:", sentences)
print("\nInFiltered Words (Stopword Remoyal):", filtered_words) 
print ("\nInStemmed Words:", stemmed_words)
print ("\nInLemmatized Words:" , lemmatized_words)