#Cosine similarity focuses on direction rather than length or amount of the document  , so it is widely used in the comparisons of texts

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "AI AI AI AI AI ",
    "AI AI "
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

similarity = cosine_similarity(X)
print(similarity)


##İki kişi kuzeye doğru yürüyor.
#Birisi 2 km yürüdü.
#Diğeri 10 km yürüdü.
#Yürüdükleri mesafe farklı.
#Ama yönleri aynı.
#Cosine Similarity de "yön" ile ilgilendiği için benzerlik 1 olur.