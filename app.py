from flask import Flask, request, render_template
from chatbot_model import load_model, predict
import csv

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

    return {'response': response}

if __name__ == "__main__":
    app.run(debug=True)
