
# 🧠 Medical Assistant Chatbot

A smart AI-powered chatbot that predicts possible medical conditions based on user-described symptoms and provides recommendations. Built using a GRU model, Naive Bayes fallback, Flask backend, and live Wikipedia integration.

---

## 🚀 Features

- ✅ Symptom-based disease prediction using GRU (PyTorch)
- ✅ Local CSV knowledge base with condition descriptions and recommendations
- ✅ Fallback to Naive Bayes if model confidence is low
- ✅ Wikipedia API integration for unknown symptoms
- ✅ Friendly responses with structured medical advice
- ✅ Flask-based web interface with minimal UI

---

## 📂 Project Structure

```
Medical_Assistant_Chatbot/
│
├── model/
│   ├── gru_model.pt         # Trained GRU model
│   ├── vocab.json           # Vocabulary for token mapping
│   └── labels.json          # Label index mapping
│
├── templates/
│   └── index.html           # Frontend interface
│
├── medical_queries.csv      # Main dataset with queries, descriptions, recommendations
├── app.py                   # Flask app logic and chatbot integration
├── chatbot_model.py         # GRU model class, prediction, and utilities
└── README.md                # Project documentation
```

---

## 🧠 Model Details

### GRU Classifier:
- Input: Tokenized user message
- Embedding dimension: `50`
- Hidden size: `64`
- Output: Predicted intent (medical condition)

### Fallback:
- **Naive Bayes** used for simple greeting or unknown queries  
- **Wikipedia API** used to fetch real-time data if condition not found in CSV

---

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Fizza-Ashfaq/Medical_Assistant_Chatbot.git
cd Medical_Assistant_Chatbot
```

### 2. Install Dependencies
```bash
pip install flask torch requests
```

### 3. Run the App
```bash
python app.py
```

Access it at: `http://127.0.0.1:5000`

---

## 💬 Example Use Cases

- **User Input**: `"I have chest pain and shortness of breath"`  
  → **Response**: `"Predicted Condition: Heart Attack"`  
  → **Advice**: `"Seek emergency medical help immediately."`

- **User Input**: `"Hi"`  
  → **Response**: `"Hi! How can I assist you today? 😊"`

---

## 📌 Limitations

- Not a replacement for professional medical diagnosis
- Dataset is small and not exhaustive
- No confidence score for prediction yet

---

## 🔮 Future Enhancements

- Add confidence threshold for GRU predictions
- Expand symptom and disease dataset
- Multilingual input handling
- Speech-to-text input
- Improve frontend interface

---

## 🤝 Contributors

- **Fizza Ashfaq**  
- **Iman Fatima** (Model integration and backend enhancements)

---

## 📜 License

This project is for educational use. For commercial use, please contact the authors.

---
