


from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "drinks", "milk"],
    ["the", "dog", "drinks", "milk"],
    ["the", "cat", "eats", "fish"],
    ["the", "dog", "eats", "meat"]
]

model = Word2Vec(sentences, vector_size=100, window=2, min_count=1,sg=1,epochs=100)
#epochs refers to the number ofhow many  times the model was trained on the sentences
#Bigger vector size does not automatically mean better embeddings.


vector = model.wv["cat"]
similarity = model.wv.similarity("cat", "dog")

print(vector)
print("\n Similarity Rate : ")
print(similarity)

print("\n   ")
print(model.wv.most_similar("cat"))


# Because of the fact that our corpus is tiny , the similarity rate between wrods "cat" and "dog"are low.

# If the corpus only contains four sentences, repeating those four sentences(epochs) doesn't magically turn them into a large and diverse corpus.