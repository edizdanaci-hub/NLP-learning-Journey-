from nltk.corpus import brown
from gensim.models import Word2Vec


sentences= [
    [word.lower() for word in sentence]
    for sentence in brown.sents()

]
print(sentences[:3])

model = Word2Vec(sentences,vector_size=100,window = 5,min_count=5,sg=1,negative=5,epochs=5)

print("Vocabulary Size:", len(model.wv))
print("Vector Size:",len(model.wv["language"]))

#semantic similarity not = dictionary synonym




print(model.wv.most_similar("language",topn=10))
print(model.wv.most_similar("language",topn=20))

for word in ["king","queen","man","woman"]:
    print(word,word in  model.wv.key_to_index)


#Embedding space'te bazı relational patterns approximately correspond to vector directions.


result = model.wv.most_similar(
    positive=["king","woman"],
    negative=["man"],
    topn=10
)

print(result)