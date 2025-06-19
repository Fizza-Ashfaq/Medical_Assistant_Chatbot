import requests
import csv

SYMPTOM_LIST = [
    'fever', 'cough', 'headache', 'sore throat',
    'nausea', 'chest pain', 'breathing difficulty',
    'dizziness', 'fatigue', 'rash', 'abdominal pain',
    'vomiting', 'diarrhea', 'back pain', 'joint pain'
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

        summary = get_medlineplus_summary(symptom)
        if not summary:
            summary = get_wikipedia_summary(symptom)

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
