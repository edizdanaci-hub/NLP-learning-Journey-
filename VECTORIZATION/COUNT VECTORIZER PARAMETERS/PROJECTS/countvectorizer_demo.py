from pygments.lexers.sql import re_message
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "Artificial Intelligence is changing the world",
    "Machine Learning is a branch of Artificial Intelligence",
    "Deep Learning is transforming AI applications",
    "Artificial Intelligence and Machine Learning are exciting"
]

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)


vectorizer2 = CountVectorizer()
X2= vectorizer2.fit_transform(documents)

removed_words = set(vectorizer2.get_feature_names_out()) - set(vectorizer.vocabulary_)


word_counts = X.toarray().sum(axis=0)




print("====VOCABULARY====")

print("Vocabulary without stopwords : ",vectorizer.get_feature_names_out())
print((len(vectorizer.get_feature_names_out())),"words")



print("--------------")

print("Original Vocabuary : ",vectorizer2.get_feature_names_out())
print((len(vectorizer2.get_feature_names_out())),"words")

print("--------------")


print("Removed words : ",removed_words)

print("====BOW MATRIX====")
print(X.toarray())
print("---------")
print(X2.toarray())


print("====WORD FREQUENCIES====")

for word , count in zip(vectorizer.get_feature_names_out(),word_counts):

    print(f"{word}:{count}")


