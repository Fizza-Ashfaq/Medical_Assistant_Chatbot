import csv
import re
import math
from collections import defaultdict, Counter

def load_dataset(path):
    texts, labels, descriptions = [], [], {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['query'].lower())
            labels.append(row['intent'])
            descriptions[row['query'].lower()] = row['description']
    return texts, labels, descriptions

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def train_naive_bayes(texts, labels):
    vocab = set()
    label_word_counts = defaultdict(Counter)
    label_counts = Counter(labels)

    for text, label in zip(texts, labels):
        tokens = tokenize(text)
        vocab.update(tokens)
        label_word_counts[label].update(tokens)

    model = {
        'label_counts': label_counts,
        'label_word_counts': label_word_counts,
        'vocab': vocab,
        'total_docs': len(labels)
    }
    return model

def predict_naive_bayes(model, text):
    user_input = text.lower().strip()

    if user_input in ["hi", "hello", "hey"]:
        return "Hi! How can I assist you today? 😊"
    if user_input in ["bye", "goodbye", "see you"]:
        return "Goodbye! Have a nice day! 👋"

    tokens = tokenize(user_input)
    scores = {}
    vocab_size = len(model['vocab'])

    for label in model['label_counts']:
        log_prob = math.log(model['label_counts'][label] / model['total_docs'])
        word_count = sum(model['label_word_counts'][label].values())

        for token in tokens:
            token_count = model['label_word_counts'][label][token]
            log_prob += math.log((token_count + 1) / (word_count + vocab_size))

        scores[label] = log_prob

    return max(scores, key=scores.get)

