import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
from nltk.util import bigrams
from nltk.util import trigrams
from nltk import FreqDist

text = """Artificial Intelligence (AI) is changing the way people communicate and work. Google and Microsoft invest heavily in AI research, while OpenAI develops advanced language models.

On 15 March 2026, researchers from Stanford University introduced a new Natural Language Processing system. The project improved machine translation and text summarization.

."""
tokens2 = word_tokenize(text)
tokens = word_tokenize(text.lower())
lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')
nlp = spacy.load('en_core_web_sm')

clean_words = [k for k in tokens if k not in stopwords and k.isalpha() ]
clean2 = [y for y in tokens2 if y.isalpha()]

lemma_words = [lemmatizer.lemmatize(word) for word in clean_words]

pos_tags = nltk.pos_tag(clean2)

doc = nlp(text)

bigrams = list(bigrams(lemma_words[:10]))
trigrams = list(trigrams(lemma_words[:10]))

freq = FreqDist(lemma_words)

types = len(set(lemma_words))
tokens = len(lemma_words)
ttr = types/tokens






print("\nCLEANED WORDS ")
print(clean_words)


print("\nPOS TAGS")

for word, tag in pos_tags:
    print(word,":",tag)

print("\nNAMED ENTITIES")

for ent in doc.ents:
    print(ent.text,"->",ent.label_)

print("\nBIGRAMS & TRIGRAMS")

for bg in bigrams:
    print(bg)

print("\n=======")
for tg in trigrams:
    print(tg)

print("\nTOP 5 WORDS")
for word , count in freq.most_common(5):
    print(word,"->",count)

print("\nTYPE TOKEN RATIO")
print(round(ttr,3))


