from sklearn.feature_extraction.text import CountVectorizer

documents = ["apple apple banana",
    "banana orange",
    "apple orange orange",
    "banana banana apple"
]

vectorizer = CountVectorizer(max_features=2)
X = vectorizer.fit_transform(documents)

print(X.toarray())
print(vectorizer.get_feature_names_out())

word_counts = X.toarray().sum(axis=0)


print("FREQUENCIES")
for word, count in zip(vectorizer.get_feature_names_out(), word_counts):
    print(f"{word}: {count}")


