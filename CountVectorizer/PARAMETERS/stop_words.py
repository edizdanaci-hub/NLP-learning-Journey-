from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "The cat and the dog",
    "The dog is happy"
]

vectorizer = CountVectorizer(stop_words='english')
x = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(x.toarray())