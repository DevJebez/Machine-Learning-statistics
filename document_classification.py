import numpy as np
import math
documents = [
    "I love programming in python",
    "Python is excellent language",
    "Machine learning is fascinating",
    "I hate bugs in my code",
    "I hate runtime errors"
]

labels = [1,1,1,0,0]


# building vocabulary 
def tokenize(doc):
    return doc.lower().split()

vocab = set()
for i in documents:
    vocab.update(tokenize(i))
vocab = sorted(vocab)
vocab_index = {word : idx for idx, word in enumerate(vocab)}

word_count = {
    0 : np.zeros(len(vocab)),
    1 : np.zeros(len(vocab))
}

class_count = {0:0 , 1:1}

print(word_count)

print(f"voacb index:{vocab_index}")

for doc, label in zip(documents,labels):
    for word in tokenize(doc):
        word_idx = vocab_index[word]
        word_count[label][word_idx] += 1
        class_count[label] += 1

# then we are going to calculate prior probability 

# prior probability  = (number of class documents / total number of documents)

total_docs = len(documents)
class_prior_probability = {
  0: labels.count(0)/total_docs,
  1 :labels.count(1)/total_docs
}

# prediction function
def predict(doc):
    tokens = tokenize(doc)
    scores = {}
    for cls in [0,1]:
        log_prob_cls = math.log(class_prior_probability[cls])
        total_words_in_cls = class_count[cls]
        for i in tokens:
            if i in vocab_index:
                idx = vocab_index[i]
                count_in_cls = word_count[cls][idx]
                prob_word_given_cls = (count_in_cls+1) / (total_words_in_cls + len(vocab))
                log_prob_cls += math.log(prob_word_given_cls)
        scores[cls] = log_prob_cls
    return 1 if scores[1] > scores[0] else 0


test_sentences = [
    "I love debugging",
    "I hate python",
    "Errors are frustrating"
]

for i in test_sentences:
    result = predict(i)
    label = "positive" if result == 1 else "Negative"
    print(f"{i} -> {label}")