import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embeddings = {}

with open('glove.6B.50d.txt',encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        embeddings[word] = vector

print("===ROYALTY EXPERIMENT===")
print("Vocabulary Size:",len(embeddings))

king_vector = embeddings["king"]
print(king_vector)

print("Vector size:",king_vector.shape)

king = embeddings["king"]
queen = embeddings["queen"]
print("King:",king)
print("Queen:",queen)

similarity=cosine_similarity(
    [king],[queen]
)

print("\nSimilarity rate between king and queen:",similarity)


car = embeddings["car"]

similarity2 = cosine_similarity(
    [car],[king]
)

print("\nSimilarity rate between car and king:",similarity2)



man = embeddings["man"]
woman = embeddings["woman"]

result = king-man +woman

similarities=cosine_similarity(
    [result],
    [king,queen,man,woman]
)

print("\nSimilarity with king,queen,man,woman:",similarities)

words= list(embeddings.keys())

embedding_matrix= np.array(list(embeddings.values()))
print("\nEmbedding matrix shape:",embedding_matrix.shape)


similarities2=cosine_similarity(
    [result],
    embedding_matrix
)[0]

top_indices=similarities2.argsort()[-10:][::-1]

for index in top_indices:
    print(words[index],similarities2[index])
#Results of this code are in the surroundings of the royalty / family / gender-related contexts they belong in the same semantic neighborhood.
#That's what we call a vector space.
print("\n===CAPITAL EXPERIMENT===")

paris = embeddings["paris"]
france = embeddings["france"]
italy=embeddings["italy"]

result= paris-france + italy
similarities3=cosine_similarity(
    [result],
    embedding_matrix
)[0]

top_indices=similarities3.argsort()[-10:][::-1]
#The model doesn't understand of the country and the capital relationship consciously. Instead, it learns the representation from the distributional statistics.
for index in top_indices:
    print(words[index],similarities3[index])

