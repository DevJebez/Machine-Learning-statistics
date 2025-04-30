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
    "I love python",
    "I hate bugs",
    "Python is fascinating",
    "Errors in code"
]
true_labels = [1, 0, 1, 0]

predicted_labels = [predict(i) for i in test_sentences]

from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix,f1_score
print("\nEvaluation metrics :\n")
print("Confusion Matrix:\n", confusion_matrix(true_labels, predicted_labels))
print("Accuracy:", accuracy_score(true_labels, predicted_labels))
print("Precision:", precision_score(true_labels, predicted_labels))
print("Recall:", recall_score(true_labels, predicted_labels))
print("F1 Score:", f1_score(true_labels, predicted_labels))