import torch
import torch.nn as nn
import json, re, csv

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

def preprocess(text):
    return re.findall(r'\b\w+\b', text.lower())

def load_model():
    with open('model/vocab.json') as f:
        vocab = json.load(f)
    with open('model/labels.json') as f:
        idx2label = json.load(f)
    idx2label = {int(k): v for k, v in idx2label.items()}

    model = GRUClassifier(len(vocab), 50, 64, len(idx2label))
    model.load_state_dict(torch.load('model/gru_model.pt', map_location='cpu'))
    model.eval()
    return model, vocab, idx2label

def get_info_and_advice(predicted_label):
    with open('medical_queries.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['query'].lower() == predicted_label.lower():
                return row['description'], row['recommendation']
    return "No info found.", "No advice available."

def predict(text, model, vocab, idx2label):
    tokens = preprocess(text)
    indices = torch.tensor([[vocab.get(token, vocab['<UNK>']) for token in tokens]])
    with torch.no_grad():
        logits = model(indices)
        pred = torch.argmax(logits, dim=1).item()
    
    predicted_label = idx2label[pred]
    description, recommendation = get_info_and_advice(predicted_label)
    
    return predicted_label, description, recommendation

if __name__ == "__main__":
    model, vocab, idx2label = load_model()
    user_input = input("📝 Describe your symptoms: ")
    label, desc, advice = predict(user_input, model, vocab, idx2label)
    
    print(f"\n🩺 Predicted Condition: {label}")
    print(f"📚 Summary: {desc}")
    print(f"💡 Advice: {advice}")
