from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "red red blue",
    "blue green",
    "green green red"
]

vectorizer = CountVectorizer(binary=True)
X=vectorizer.fit_transform(documents)

print(vectorizer.vocabulary_)
print(vectorizer.get_feature_names_out())
print(X.toarray())



