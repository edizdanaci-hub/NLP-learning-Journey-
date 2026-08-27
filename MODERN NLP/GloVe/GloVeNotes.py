# GloVe = Global Vectors for word Representation, it mostly uses the co-occurace stats of the words

#Word2Vec = "Bu kelimenin çevresinde ne var?" ,GloVe = "Corpus'un tamamında kelimeler birbirleriyle nasıl birlikte görülüyor?"

#If two words co-occure more than expected we would ecpect the PMI(Pointwise mutual Information) to be high , if it is close to teh expected PMI = 0 ,If it is seen less than we expect PMI will be low


#Co_occurrence Ratios
# GloVe says we should check the importance of the Co-occurrence rate with a weightening function
# J = weighted error between model and log(Xij)

