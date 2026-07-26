from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "Cat cat CAT",
    "CAT",
    "cat"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)


print(vectorizer.get_feature_names_out())
print (X.toarray())
