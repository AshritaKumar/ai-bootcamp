import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("movie_reviews")
nltk.download('punkt_tab')

# Example text
text = "NLTK is a great library for Natural Language Processing! It helps in tokenization, removing stopwords, and more."

# 1. Tokenize
tokens = word_tokenize(text)
print("Tokens:", tokens)

# 2. Lowercase
tokens = [t.lower() for t in tokens]

# 3. Remove punctuation
tokens = [t for t in tokens if t not in string.punctuation]

# 4. Remove stopwords (English)
stop_words = set(stopwords.words("english"))
tokens = [t for t in tokens if t not in stop_words]

print("After stopword removal:", tokens)

# 5. Using different corpora
# Stopwords in Spanish
stop_words_es = set(stopwords.words("spanish"))
print("\nSpanish Stopwords (sample):", list(stop_words_es)[:10])

# WordNet example (synonyms)
from nltk.corpus import wordnet
syns = wordnet.synsets("happy")
print("\nSynonyms for 'happy':", [s.lemma_names()[0] for s in syns[:5]])

# Movie reviews corpus example
from nltk.corpus import movie_reviews
print("\nNumber of movie reviews:", len(movie_reviews.fileids()))
print("First review words:", movie_reviews.words(movie_reviews.fileids()[0])[:20])
