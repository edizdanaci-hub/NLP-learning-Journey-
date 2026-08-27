# we will face some conflicts such as positive and negative examples when we start working with larger corpuses, which include millions of words in them.
#We try to teach the model whether those two words have the same context relationship or not.

from gensim.models import Word2Vec


sentences = [ ["the", "cat", "drinks", "milk"],
    ["the", "dog", "drinks", "milk"],
    ["the", "cat", "eats", "fish"],
    ["the", "dog", "eats", "meat"]
]


model2 = Word2Vec(
    sentences,
    vector_size=100,
    window=2,
    min_count=1,
    sg=1,
    negative=5,
    epochs=100
)


# We want the model to use approximately 5 negative samples for every training example.
# when we alter negative to 0 (negative = 0) Model doesn't use any negative samples.

print(model2.wv.similarity("cat","dog"))