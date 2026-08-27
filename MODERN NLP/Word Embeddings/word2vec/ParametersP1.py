from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "drinks", "milk"],
    ["the", "dog", "drinks", "milk"],
    ["the", "cat", "eats", "fish"],
    ["the", "dog", "eats", "meat"]
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=2,
    min_count=1,
    workers=4,
    sg=0
)

# sg=0 for CBOW and sg = 1 for Skip-Gram
vector = model.wv["cat"]
similarity = model.wv.similarity("cat", "dog")

print(len(vector))
print(similarity)
print(model.wv.most_similar("cat"))