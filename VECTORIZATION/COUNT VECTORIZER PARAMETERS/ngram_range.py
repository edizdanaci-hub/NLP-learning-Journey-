from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "deep learning improves models"
]

vectorizer = CountVectorizer(
    ngram_range=(1,2)
)
x = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(x.toarray())