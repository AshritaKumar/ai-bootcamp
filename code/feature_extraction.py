from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["I love NLP", "NLP is fun", "I love Python"]
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(docs)
print(features.toarray())
print(vectorizer.get_feature_names_out())
