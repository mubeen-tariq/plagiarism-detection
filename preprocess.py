import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()


    tokens = word_tokenize(text)


    tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]


    tokens = [lemmatizer.lemmatize(word) for word in tokens]


    processed_text = ' '.join(tokens)

    return processed_text
