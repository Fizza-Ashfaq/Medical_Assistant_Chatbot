import requests
import csv

SYMPTOM_LIST = [
    "fever", "chills", "fatigue", "weight loss", "weight gain", "loss of appetite", "night sweats", "general weakness",
    "cough", "shortness of breath", "wheezing", "chest pain", "runny nose", "nasal congestion", "sore throat", "sneezing",
    "abdominal pain", "nausea", "vomiting", "diarrhea", "constipation", "bloating", "heartburn", "indigestion", "bloody stool",
    "headache", "dizziness", "fainting", "seizures", "confusion", "numbness", "tingling", "memory loss", "speech difficulty",
    "rash", "itching", "dry skin", "acne", "hives", "bruising", "skin discoloration", "eczema", "psoriasis", "blisters",
    "joint pain", "muscle pain", "back pain", "swollen joints", "stiffness", "limited mobility",
    "frequent urination", "painful urination", "blood in urine", "urinary incontinence", "difficulty urinating",
    "chest tightness", "palpitations", "irregular heartbeat", "high blood pressure", "low blood pressure",
    "irregular periods", "heavy bleeding", "menstrual cramps", "pelvic pain", "erectile dysfunction", "infertility",
    "depression", "anxiety", "panic attacks", "mood swings", "insomnia", "restlessness", "irritability", "hallucinations",
    "blurred vision", "double vision", "eye pain", "dry eyes", "watery eyes", "vision loss", "red eyes",
    "ear pain", "hearing loss", "ringing in ears", "nosebleeds", "hoarseness", "loss of smell", "loss of taste",
    "malaria", "dengue", "tuberculosis", "influenza", "measles", "mumps", "rubella", "chickenpox", "hepatitis", "covid-19",
    "diabetes", "hypertension", "asthma", "arthritis", "cancer", "alzheimer's", "parkinson's", "epilepsy", "crohn's disease",
    "lupus", "thyroid disorders", "anemia", "migraine", "pneumonia", "bronchitis", "ulcer", "kidney stones", "gallstones",
    "urinary tract infection", "fibromyalgia", "gout", "osteoporosis", "hiv", "autism", "adhd", "celiac disease", "pcos", "endometriosis"
]

def get_medlineplus_summary(symptom):
    try:
        rss_url = f'https://medlineplus.gov/{symptom.replace(" ", "")}.html'
        return f"You can read about {symptom} here: {rss_url}"
    except:
        return None

def get_wikipedia_summary(symptom):
    base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    url = base_url + symptom.replace(" ", "%20")
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("extract", "")
    return None

def fetch_symptom_descriptions(symptom_list):
    data = []

    for symptom in symptom_list:
        print(f"Fetching info for: {symptom}")

        summary = get_wikipedia_summary(symptom)
        if not summary:
            summary = get_medlineplus_summary(symptom)

        if summary:
            data.append({'query': symptom, 'intent': 'symptom_info', 'description': summary})
        else:
            print(f"Failed to find data for: {symptom}")

    with open('medical_queries.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['query', 'intent', 'description'])
        writer.writeheader()
        writer.writerows(data)

    print("Data saved to medical_queries.csv")

if __name__ == "__main__":
    fetch_symptom_descriptions(SYMPTOM_LIST)
