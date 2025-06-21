from flask import Flask, request, render_template
from chatbot_model import load_model, predict
import csv
from flask import Flask, render_template, request
import csv
import re
import math
import requests
from collections import defaultdict, Counter

app = Flask(__name__)
model, vocab, idx2label = load_model()

# Load symptom descriptions and recommendations
info_data = {}
with open('medical_queries.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        info_data[row['query'].lower()] = {
            "description": row['description'],
            "recommendation": row.get('recommendation', 'No advice available.')
        }

### === Load Dataset Functions === ###
def load_dataset(path):
    texts, labels, descriptions, treatments = [], [], {}, {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row['query'].lower()
            texts.append(query)
            labels.append(row['intent'])
            descriptions[query] = row['description']
            treatments[query] = row.get('treatment', '')
    return texts, labels, descriptions, treatments

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


### === Wikipedia Fetching Functions === ###
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

### === Improved Keyword Extraction and Mapping === ###
def extract_main_symptom(user_input):
    keywords = [
        "fever", "cough", "headache", "fracture", "bone fracture", "covid", "covid-19", 
        "diabetes", "malaria", "cold", "flu", "injury", "infection", "asthma", "hypertension"
    ]
    for word in keywords:
        if word in user_input.lower():
            return word
    return user_input.strip()  # fallback to full input if no keyword matches

keyword_mapping = {
    "covid": "COVID-19",
    "covid-19": "COVID-19",
    "fracture": "Bone fracture",
    "bone fracture": "Bone fracture",
    "cold": "Common cold",
    "flu": "Influenza",
    "diabetes": "Diabetes mellitus",
}

### === Initialize Chatbot Model === ###
texts, labels, descriptions, treatments = load_dataset('medical_queries.csv')
model = train_naive_bayes(texts, labels)

### === Flask Routes === ###
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    predicted_label, description, recommendation = predict(user_input, model, vocab, idx2label)

    # fallback if predict misses
    info = info_data.get(predicted_label.lower(), {
        "description": description,
        "recommendation": recommendation
    })

    response = f"""
🤖 Medical Assistant Bot  
─────────────────────────────  
📝 You said:  
   ➤ "{user_input.strip()}"  

🩺 Diagnosis Prediction:  
   ➤ {predicted_label.capitalize()}  

📚 About {predicted_label.capitalize()}:  
   {info['description']}  

💡 What You Should Do:  
   ➤ {info['recommendation']}  

🩻 Wishing you a quick recovery! ❤️
""".strip()
    intent_or_response = predict_naive_bayes(model, user_input)

    if intent_or_response in ["Hi! How can I assist you today? 😊", "Goodbye! Have a nice day! 👋"]:
        response = intent_or_response
    else:
        # Check for match in local CSV data first
        matched = None
        treatment = None
        for query in descriptions:
            if query in user_input.lower():
                matched = descriptions[query]
                treatment = treatments.get(query, '')
                break
            elif any(word in user_input.lower().split() for word in query.split()):
                matched = descriptions[query]
                treatment = treatments.get(query, '')
                break

        if matched:
            response = matched
            if treatment:
                response += f"\n\n**Treatment:** {treatment}"
        else:
            # Use improved keyword extraction and Wikipedia fetching
            keyword = extract_main_symptom(user_input)
            keyword = keyword_mapping.get(keyword, keyword)  # Map to correct Wikipedia title if available

            description, treatment = get_wikipedia_sections(keyword)

            if description:
                response = description
                if treatment:
                    response += f"\n\n**Treatment:** {treatment}"
                response += "\n\n(This info is fetched live from Wikipedia)"
            else:
                response = "I'm still learning! 🤖 Try asking a more direct medical question or rephrasing it."

    return {'response': response}

if __name__ == "__main__":
    app.run(debug=True)
