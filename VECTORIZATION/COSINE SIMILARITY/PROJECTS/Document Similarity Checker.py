from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import re

def get_documents():
    text1 = input("Enter the first text : \n")
    text2 = input("\nEnter the second text : ")
    return text1, text2

document1, document2 = get_documents()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "",text)
    return text

def calculate_similarity(text1, text2):
    documents = [text1, text2]
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(documents)
    similarity=cosine_similarity(x)
    return similarity[0][1]

def interpret_similarity(score):
    if score >=0.80:
        return "These documents are highly similar."
    elif score >=0.50:
        return "These documents are moderately similar."
    else:
        return "These documents are not very similar.."




document1 = clean_text(document1)
document2 = clean_text(document2)

print("-" * 40)

print()
print("Document 1")
print(document1)

print()
print("Document 2")
print(document2)

print()
score = calculate_similarity(document1, document2)
message = interpret_similarity(score)

print(f"\nSimilarity score : {score:.2%}")
print(message)

