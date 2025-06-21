from flask import Flask, request, render_template
from chatbot_model import load_model, predict
import re, csv, math, requests
from collections import defaultdict, Counter

app = Flask(__name__)
gru_model, vocab, idx2label = load_model()

### === CSV Loader === ###
def load_dataset(path):
    texts, labels, descriptions, treatments = [], [], {}, {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row['query'].lower()
            texts.append(query)
            labels.append(row['intent'])
            descriptions[query] = row['description']
            treatments[query] = row.get('recommendation', '')
    return texts, labels, descriptions, treatments

texts, labels, descriptions, recommendations = load_dataset('medical_queries.csv')

### === Tokenizer and Naive Bayes === ###
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

    return {
        'label_counts': label_counts,
        'label_word_counts': label_word_counts,
        'vocab': vocab,
        'total_docs': len(labels)
    }

def predict_naive_bayes(model, text):
    text = text.lower().strip()
    if text in ["hi", "hello", "hey"]:
        return "Hi! How can I assist you today? 😊"
    if text in ["bye", "goodbye", "see you"]:
        return "Goodbye! Take care! 👋"

    tokens = tokenize(text)
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

nb_model = train_naive_bayes(texts, labels)

### === Wikipedia Fallback === ###
def get_wikipedia_sections(symptom):
    url = f"https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{symptom.replace(' ', '%20')}"
    response = requests.get(url)
    if response.status_code != 200:
        return None, None

    data = response.json()
    description = data.get("lead", {}).get("sections", [{}])[0].get("text", "")
    treatment = ""
    for section in data.get("sections", []):
        title = section.get("line", "").lower()
        if "treatment" in title or "management" in title:
            treatment = section.get("text", "").replace('<p>', '').replace('</p>', '').strip()
            break
    return description.strip(), treatment.strip()

### === Keyword Mapping === ###
keyword_mapping = {
    "covid": "COVID-19",
    "covid-19": "COVID-19",
    "fracture": "Bone fracture",
    "cold": "Common cold",
    "flu": "Influenza",
    "diabetes": "Diabetes mellitus",
}

def extract_main_symptom(user_input):
    keywords = [
        "fever", "cough", "headache", "fracture", "covid", "covid-19", 
        "diabetes", "malaria", "cold", "flu", "injury", "infection", "asthma", "hypertension"
    ]
    for word in keywords:
        if word in user_input.lower():
            return keyword_mapping.get(word, word)
    return user_input.strip()

### === Flask Routes === ###
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    
    # Basic intent check
    basic_reply = predict_naive_bayes(nb_model, user_input)
    if basic_reply.startswith("Hi!") or basic_reply.startswith("Goodbye"):
        return {'response': basic_reply}

    # GRU model prediction
    predicted_label, description, recommendation = predict(user_input, gru_model, vocab, idx2label)

    # Check if we have local data
    desc = descriptions.get(predicted_label.lower(), description)
    rec = recommendations.get(predicted_label.lower(), recommendation)

    if not desc or desc.strip() == "":
        # Try Wikipedia as fallback
        keyword = extract_main_symptom(user_input)
        desc, rec = get_wikipedia_sections(keyword)
        if desc:
            response = f"""
🤖 Medical Assistant Bot  
─────────────────────────────  
📝 You said:  
   ➤ "{user_input.strip()}"  

📚 {keyword.capitalize()} Summary:  
   {desc}  

💡 Recommended Steps:  
   ➤ {rec or 'No advice found.'}  

🌐 (Info fetched from Wikipedia)  
""".strip()
        else:
            response = "I'm still learning! Try asking a more specific symptom or rephrasing the question."
    else:
        # Build structured response
        response = f"""
🤖 Medical Assistant Bot  
─────────────────────────────  
📝 You said:  
   ➤ "{user_input.strip()}"  

🩺 Predicted Condition:  
   ➤ {predicted_label.capitalize()}  

📚 About {predicted_label.capitalize()}:  
   {desc}  

💡 What You Should Do:  
   ➤ {rec}  

🩻 Wishing you a quick recovery! ❤️
""".strip()

    return {'response': response}

if __name__ == "__main__":
    app.run(debug=True)
