import requests
import csv
import time

SYMPTOM_LIST = [
    "fever", "fatigue", "headache", "cough", "diabetes", "asthma", "anxiety", "depression",
    "arthritis", "migraine", "pneumonia", "ulcer", "eczema", "lupus", "covid-19", "parkinson's disease",
    "alzheimer's disease", "thyroid disease", "hypertension", "anemia", "autism", "ADHD", "PCOS", "endometriosis"
]

def get_wikipedia_summary(symptom):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{symptom.replace(' ', '%20')}"
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            return data.get("extract", None)
        return None
    except Exception as e:
        print(f"⚠️ Error for {symptom}: {e}")
        return None

def fetch_and_save():
    data = []

    for symptom in SYMPTOM_LIST:
        print(f"🔍 Fetching: {symptom}")
        summary = get_wikipedia_summary(symptom)
        time.sleep(0.5)

        if summary:
            recommendation = f"For {symptom}, it is advised to consult a healthcare provider. Rest, hydration, and prescribed medications are typically important."
            data.append({
                "query": symptom,
                "intent": "symptom_info",
                "description": summary,
                "recommendation": recommendation
            })
        else:
            print(f"❌ No info found for {symptom}")

    with open("medical_queries.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["query", "intent", "description", "recommendation"])
        writer.writeheader()
        writer.writerows(data)

    print(f"✅ Saved {len(data)} entries to medical_queries.csv")

if __name__ == "__main__":
    fetch_and_save()
