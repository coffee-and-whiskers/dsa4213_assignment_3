import os
import json
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def process_huatuo_dataset(sample_size):
    ds = load_dataset("shibing624/huatuo_medical_qa_sharegpt")
    train_data = ds['train']

    if len(train_data) > sample_size:
        sampled_data = train_data.shuffle(seed=42).select(range(sample_size))
    else:
        sampled_data = train_data

    processed_data = []
    for item in sampled_data:
        conversations = item.get('conversations', [])
        human_msg = ""
        assistant_msg = ""

        for msg in conversations:
            if msg.get('from') == 'human':
                human_msg = msg.get('value', '')
            elif msg.get('from') == 'gpt':
                assistant_msg = msg.get('value', '')

        if human_msg and assistant_msg:
            processed_data.append({
                'human': human_msg,
                'assistant': assistant_msg
            })

    return processed_data

def save_jsonl(data, output_dir, split_name):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f'huatuo_{split_name}.jsonl')

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def main():
    sample_size = 30366
    data = process_huatuo_dataset(sample_size)

    # split train test
    train_data, test_data = train_test_split(
        data,
        test_size=0.1,
        random_state=42
    )

    # save splits
    output_dir = 'data/huatuo_formatted'
    save_jsonl(train_data, output_dir, 'train')
    save_jsonl(test_data, output_dir, 'test')

    print(f"huatuo: {len(data)} total, {len(train_data)} train, {len(test_data)} test")

if __name__ == "__main__":
    main()
