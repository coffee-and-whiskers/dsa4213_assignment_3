import os
import json
import re
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text, patterns):
    if not isinstance(text, str):
        return ""
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', cleaned).strip()

def process_ai_medical_chatbot():
    ds = load_dataset("ruslanmv/ai-medical-chatbot")
    train_df = pd.DataFrame(ds['train'])

    filtered_df = train_df[
        (train_df['Description'].str.contains('Q.', case=False, na=False)) &
        (
            (train_df['Doctor'].str.contains('Hello.', case=False, na=False)) |
            (train_df['Doctor'].str.contains('Hi.', case=False, na=False)) |
            (train_df['Doctor'].str.contains('Hello,', case=False, na=False))
        ) &
        (
            (train_df['Patient'].str.contains('Hi doctor,', case=False, na=False)) |
            (train_df['Patient'].str.contains('Hello doctor,', case=False, na=False))
        )
    ]

    description_patterns = [r'Q\.\s*']
    doctor_patterns = [r'^Hello\.\s*', r'^Hi\.\s*', r'^Hello,\s*']
    patient_patterns = [r'^Hi doctor,\s*', r'^Hello doctor,\s*']

    processed_data = []
    for idx, row in filtered_df.iterrows():
        description = clean_text(row.get('Description', ''), description_patterns)
        patient = clean_text(row.get('Patient', ''), patient_patterns)
        doctor = clean_text(row.get('Doctor', ''), doctor_patterns)

        if description and patient and doctor:
            processed_data.append({
                'human': f"{description} {patient}".strip(),
                'assistant': doctor
            })

    return processed_data

def save_jsonl(data, output_dir, split_name):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f'aimc_{split_name}.jsonl')

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def main():
    data = process_ai_medical_chatbot()

    # split train test
    train_data, test_data = train_test_split(
        data,
        test_size=0.1,
        random_state=42
    )

    # save splits
    output_dir = 'data/aimc_formatted'
    save_jsonl(train_data, output_dir, 'train')
    save_jsonl(test_data, output_dir, 'test')

    print(f"aimc: {len(data)} total, {len(train_data)} train, {len(test_data)} test")

if __name__ == "__main__":
    main()
