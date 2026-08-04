from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "Artificial intelligence is changing the world",
    "Machine learning is a branch of artificial intelligence",
    "Deep learning and artificial intelligence are exciting",
    "AI is transforming the world with machine learning"
]
vectorizer1 = CountVectorizer()
vectorizer2 = CountVectorizer(stop_words='english')
vectorizer3 =  CountVectorizer(max_features=2)
vectorizer4 = CountVectorizer(binary=True)
vectorizer5 = CountVectorizer(min_df=2)
vectorizer6 = CountVectorizer(max_df=2)
vectorizer7 = CountVectorizer(ngram_range=(1,2))

x = vectorizer1.fit_transform(documents)
x2 = vectorizer2.fit_transform(documents)
x3 = vectorizer3.fit_transform(documents)
x4 = vectorizer4.fit_transform(documents)
x5 = vectorizer5.fit_transform(documents)
x6 = vectorizer6.fit_transform(documents)
x7 = vectorizer7.fit_transform(documents)



print("=" * 40)
print("DEFAULT COUNT VECTORIZER")
print("=" * 40)

print("\nVocabulary:",vectorizer1.get_feature_names_out())
print("\nBoW Matrix:",x.toarray())

print("=" * 40)
print("STOP WORDS")
print("=" * 40)

print("\nVocabulary:",vectorizer2.get_feature_names_out())
print("\nBoW Matrix:",x2.toarray())


print("=" * 40)
print("MAX FEATURES")
print("=" * 40)

print("\nVocabulary:",vectorizer3.get_feature_names_out())
print("\nBoW Matrix:",x3.toarray())




print("=" * 40)
print("BINARY")
print("=" * 40)


print("\nVocabulary:",vectorizer4.get_feature_names_out())
print("\nBoW Matrix:",x4.toarray())



print("=" * 40)
print("MIN_DF & MAX_DF")
print("=" * 40)

print("\nVocabulary:",vectorizer5.get_feature_names_out())
print("\nBoW Matrix:",x5.toarray())

print("\nVocabulary:",vectorizer6.get_feature_names_out())
print("\nBoW Matrix:",x6.toarray())





print("=" * 40)
print("NGRAM RANGE")
print("=" * 40)

print("\nVocabulary:",vectorizer7.get_feature_names_out())
print("\nBoW Matrix:",x7.toarray())

print("=" * 40)
print("VOCABULARY SIZE ")
print("=" * 40)

print("Vocabulary Size:",
      len(vectorizer1.get_feature_names_out()))

