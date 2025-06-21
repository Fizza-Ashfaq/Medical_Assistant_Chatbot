import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv, json, re
from collections import Counter
from sklearn.model_selection import train_test_split
import os

# Dataset class
class SymptomDataset(Dataset):
    def __init__(self, texts, labels, vocab, label2idx):
        self.data = [torch.tensor([vocab.get(word, vocab['<UNK>']) for word in text], dtype=torch.long) for text in texts]
        self.labels = [label2idx[label] for label in labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]

# GRU Classifier
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        return self.fc(hidden[-1])

# Tokenization
def preprocess(text):
    return re.findall(r'\b\w+\b', text.lower())

# Build vocabulary
def build_vocab(texts, min_freq=1):
    counter = Counter(word for text in texts for word in text)
    vocab = {word: i+2 for i, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

# Padding function
def pad_sequences(batch, pad_idx=0):
    max_len = max(len(x[0]) for x in batch)
    padded_x = [torch.cat([x[0], torch.full((max_len - len(x[0]),), pad_idx)]) for x in batch]
    labels = [x[1] for x in batch]
    return torch.stack(padded_x), torch.tensor(labels)

if __name__ == "__main__":
    # Load user-style query data
    texts, labels = [], []
    with open('medical_queries.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(preprocess(row['query']))         # ✅ Train on user-style queries
            labels.append(row['query'].lower())            # ✅ Predict the disease/symptom name

    # Build vocab and label dictionaries
    label2idx = {l: i for i, l in enumerate(sorted(set(labels)))}
    idx2label = {i: l for l, i in label2idx.items()}
    vocab = build_vocab(texts)

    # Save vocab + labels
    os.makedirs('model', exist_ok=True)
    with open('model/vocab.json', 'w') as f:
        json.dump(vocab, f)
    with open('model/labels.json', 'w') as f:
        json.dump(idx2label, f)

    # Split and prepare DataLoader
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
    train_ds = SymptomDataset(X_train, y_train, vocab, label2idx)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=pad_sequences)

    # Setup model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUClassifier(len(vocab), 50, 64, len(label2idx)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            opt.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), 'model/gru_model.pt')
    print("✅ GRU Model trained and saved to model/gru_model.pt")
