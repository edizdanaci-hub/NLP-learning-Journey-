from sklearn.feature_extraction.text import CountVectorizer

documents = ["apple banana",
    "banana orange",
    "banana apple",
    "grape banana"
]

vectorizer = CountVectorizer(min_df=2)
x = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(x.toarray())

