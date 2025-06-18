import requests
import csv

SYMPTOM_LIST = [
    'fever', 'cough', 'headache', 'sore throat',
    'nausea', 'chest pain', 'breathing difficulty'
]

def fetch_symptom_descriptions(symptom_list):
    base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    data = []
    
    for symptom in symptom_list:
        url = base_url + symptom.replace(" ", "%20")
        response = requests.get(url)
        if response.status_code == 200:
            summary = response.json().get("extract", "")
            print(f"Fetched: {symptom}")
            data.append({'query': symptom, 'intent': 'symptom_info', 'description': summary})
        else:
            print(f"Failed: {symptom}")
    
    with open('medical_queries.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['query', 'intent', 'description'])
        writer.writeheader()
        writer.writerows(data)
    print("Data saved to medical_queries.csv")

if __name__ == "__main__":
    fetch_symptom_descriptions(SYMPTOM_LIST)
