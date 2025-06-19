from flask import Flask, render_template, request
from chatbot_model import load_dataset, train_naive_bayes, predict_naive_bayes

app = Flask(__name__)

texts, labels, descriptions = load_dataset('medical_queries.csv')
model = train_naive_bayes(texts, labels)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    intent_or_response = predict_naive_bayes(model, user_input)

    if intent_or_response in ["Hi! How can I assist you today? 😊", "Goodbye! Have a nice day! 👋"]:
        response = intent_or_response
    else:
        matched = None
        for query in descriptions:
            if query in user_input.lower():
                matched = descriptions[query]
                break
            elif any(word in user_input.lower().split() for word in query.split()):
                matched = descriptions[query]
                break

        if matched:
            response = matched
        else:
            response = "I'm still learning! 🤖 Try asking a more direct medical question or rephrasing it."

    return {'response': response}

if __name__ == '__main__':
    app.run(debug=True)
