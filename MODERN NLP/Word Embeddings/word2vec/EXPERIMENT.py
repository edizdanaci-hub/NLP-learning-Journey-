from gensim.models import Word2Vec
sentences = [
    ["the", "king", "rules", "the", "kingdom"],
    ["the", "queen", "rules", "the", "kingdom"],
    ["the", "king", "is", "a", "man"],
    ["the", "queen", "is", "a", "woman"],
    ["the", "prince", "is", "a", "young", "man"],
    ["the", "princess", "is", "a", "young", "woman"],
    ["the", "king", "and", "queen", "live", "in", "the", "castle"],
    ["the", "prince", "and", "princess", "live", "in", "the", "castle"],
    ["the", "man", "works", "in", "the", "castle"],
    ["the", "woman", "works", "in", "the", "castle"]
]

#Small corpus = unreliable embeddings.

model = Word2Vec(sentences,vector_size=100,window=2,min_count=1,sg=1,negative=5,epochs=100)

##Our corpus is still insufficient to learn the semantic structure. And because of that, the most similar words to the word king are occurring as "in", "the" ,"a".
#Word embeddings need sufficient distributional evidence.

print("==Most Similar Words==")
print(model.wv.most_similar("king"))
print(model.wv.most_similar("queen"))


print("\n==Similarity Rates==")
print("\nking-queen:")
print(model.wv.similarity("king","queen"))
print("\nman-woman:")
print(model.wv.similarity("man","woman"))
print("\nman-king:")
print(model.wv.similarity("man","king"))
print("\nqueen-woman:")
print(model.wv.similarity("queen","woman"))
