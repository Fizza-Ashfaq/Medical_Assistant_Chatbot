import requests
import csv

SYMPTOM_LIST = [
    "fever", "cough", "headache", "sore throat", "nausea", "chest pain", "breathing difficulty"
    # Add more symptoms/diseases as needed
]

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

def generate_medical_csv(symptom_list, output_file='medical_queries.csv'):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['query', 'intent', 'description', 'treatment'])
        writer.writeheader()

        for symptom in symptom_list:
            print(f"Fetching data for: {symptom}")
            description, treatment = get_wikipedia_sections(symptom)

            if description:
                writer.writerow({
                    'query': symptom,
                    'intent': 'symptom_info',
                    'description': description,
                    'treatment': treatment if treatment else "Consult a healthcare provider for treatment options."
                })
            else:
                print(f"Failed to find description for: {symptom}")

    print(f"✅ CSV generated: {output_file}")

if __name__ == "__main__":
    generate_medical_csv(SYMPTOM_LIST)
