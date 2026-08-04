#TfidfVectorizer hem kelimenin belgede ne kadar sık geçtiğini (TF) hem de tüm belgeler arasında ne kadar ayırt edici olduğunu (IDF) dikkate alarak her kelimeye bir ağırlık atar.

from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "Artificial intelligence is changing the world",
    "Machine learning is a branch of artificial intelligence",
    "Deep learning is transforming AI applications",
    "Artificial intelligence and machine learning are exciting"
]

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(x.toarray())


