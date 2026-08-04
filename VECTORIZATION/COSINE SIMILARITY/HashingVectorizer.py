# hash function turns a word into a fixed-length number .
# hash collision happens when two different words drops onto the same column
from sklearn.feature_extraction.text import HashingVectorizer

documents = [
    "Machine learning is amazing",
    "Machine learning is awesome"
]

vectorizer = HashingVectorizer(n_features=8)
# n_features parameter determines how many columns wil be constructed
#One of the main differences between HashingVectorizer and CountVectorizer is that we use transform instead of fit_transform n the former as there is no vocabulary to learn


x = vectorizer.transform(documents)
print(x.toarray())