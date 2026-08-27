## The purpose of word embeddings is to represents the words that have similar meanings close to each other mathematcally
# before we asked how many times did that word apper , now we ask To which words is the meaning of this word closer?
### !!!!"Words that appear in similar contexts tend to have similar meanings."!!!

#Distributional hyphothesis :"You shall know a word by the company it keeps." In other words A word's meaning can be inferred from the words that usually appear around it.
#One of the biggest limitations in Word2vec is that it only gives one fixed vector per word.

#Contextual embedding means to vectorize the word not only according to the meaning of the it but also the context in which it is used.


#Continuous bag of words asks the question: Given these surrounding words, which word should be in the middle?
#CBOW = I___ NLP
#Skip Grams works in the opposite direction.


#When two words appear in the same context over and over again, they are more likely to get closer in the vector space.

#In the model, king =male + royalty. The model isn't given these linguistic features beforehand. These relationships emerge from the learned vector space.


#The main difference between count vectorizer and word2vec is that with count vectorizer, the vectors essentially represent word occurrences, while in word2vec, it also shows which words are connected because they are used the same contexts.

