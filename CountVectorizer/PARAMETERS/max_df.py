from sklearn.feature_extraction.text import CountVectorizer

docuents = [ "cat dog",
    "dog bird",
    "dog fish",
    "dog cat"]

vectorizer = CountVectorizer(max_df=2)
x = vectorizer.fit_transform(docuents)



print(vectorizer.get_feature_names_out())
print(x.toarray())