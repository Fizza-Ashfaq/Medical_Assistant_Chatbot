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
    intent = predict_naive_bayes(model, user_input)

    # If intent is symptom_info, try to match
    matched = None
    for query in descriptions:
        if query in user_input.lower():
            matched = descriptions[query]
            break

    if matched:
        response = matched
    else:
        response = "I'm here to help, could you clarify your request?"

    return {'response': response}

if __name__ == '__main__':
    app.run(debug=True)
