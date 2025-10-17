
import os
import json
import torch
from tqdm import tqdm
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from peft import PeftModel
import numpy as np
from bert_score import score as bertscore_compute
import argparse
import sys
import importlib.util

# Load evaluate package correctly (avoid local file conflict)
spec = importlib.util.find_spec('evaluate')
if spec and 'site-packages' in spec.origin:
    import evaluate as eval_module
    load_metric = eval_module.load
else:
    # Fallback: use rouge_score directly
    from rouge_score import rouge_scorer
    load_metric = None


def load_test_data(jsonl_path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_model_and_tokenizer(model_path, base_model='google/mt5-base', use_lora=True):
    tokenizer = MT5Tokenizer.from_pretrained(
        model_path if not use_lora else base_model
    )

    if use_lora:
        # Load base model + LoRA adapter
        base = MT5ForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()  # Merge for faster inference
    else:
        # Load full fine-tuned model
        model = MT5ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        )

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    return model, tokenizer


def generate_answers(model, tokenizer, questions, batch_size=8, max_length=512):
    answers = []
    for i in tqdm(range(0, len(questions), batch_size), desc="generating"):
        batch_questions = questions[i:i+batch_size]
        inputs = [f"medical question: {q}" for q in batch_questions]
        encoded = tokenizer(
            inputs,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_length=max_length,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        batch_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        answers.extend(batch_answers)
    return answers


def compute_rouge(predictions, references):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    return {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores)
    }


def compute_f1(predictions, references):
    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())

        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            f1_scores.append(1.0)
            continue
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1_scores.append(0.0)
            continue

        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)

    return {'f1': np.mean(f1_scores)}


def compute_exact_match(predictions, references):
    exact_matches = []
    for pred, ref in zip(predictions, references):
        pred_normalized = pred.strip().lower()
        ref_normalized = ref.strip().lower()
        exact_matches.append(1.0 if pred_normalized == ref_normalized else 0.0)
    return {'exact_match': np.mean(exact_matches)}




def compute_bertscore_metric(predictions, references, lang='zh', model_type=None):
    if model_type is None:
        model_type = 'bert-base-chinese' if lang == 'zh' else 'bert-base-uncased'

    P, R, F1 = bertscore_compute(
        predictions,
        references,
        lang=lang,
        model_type=model_type,
        verbose=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    return {
        'bertscore_precision': P.mean().item(),
        'bertscore_recall': R.mean().item(),
        'bertscore_f1': F1.mean().item()
    }




def evaluate_model(
    model_path,
    test_jsonl_path,
    output_path,
    base_model='google/mt5-base',
    use_lora=True,
    batch_size=8,
    max_length=512,
    lang='zh',
    bertscore_model=None,
    save_predictions=True
):

    test_data = load_test_data(test_jsonl_path)
    questions = [item['human'] for item in test_data]
    references = [item['assistant'] for item in test_data]

    model, tokenizer = load_model_and_tokenizer(model_path, base_model, use_lora)
    predictions = generate_answers(model, tokenizer, questions, batch_size, max_length)

    rouge_scores = compute_rouge(predictions, references)
    f1_scores = compute_f1(predictions, references)
    bertscore_scores = compute_bertscore_metric(
        predictions, references, lang=lang, model_type=bertscore_model
    )
    em_scores = compute_exact_match(predictions, references)

    results = {
        'model_path': model_path,
        'test_data': test_jsonl_path,
        'num_samples': len(test_data),
        'metrics': {
            **rouge_scores,
            **f1_scores,
            **bertscore_scores,
            **em_scores
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"rouge1: {rouge_scores['rouge1']:.4f}")
    print(f"rouge2: {rouge_scores['rouge2']:.4f}")
    print(f"rougeL: {rouge_scores['rougeL']:.4f}")
    print(f"f1: {f1_scores['f1']:.4f}")
    print(f"bertscore_f1: {bertscore_scores['bertscore_f1']:.4f}")
    print(f"exact_match: {em_scores['exact_match']:.4f}")

    if save_predictions:
        pred_path = output_path.replace('.json', '_predictions.jsonl')
        with open(pred_path, 'w', encoding='utf-8') as f:
            for q, p, r in zip(questions, predictions, references):
                f.write(json.dumps({
                    'question': q,
                    'prediction': p,
                    'reference': r
                }, ensure_ascii=False) + '\n')

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LoRA fine-tuned model on test set')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model (LoRA adapter or full model)')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test JSONL file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save evaluation results')
    parser.add_argument('--base_model', type=str, default='google/mt5-base',
                        help='Base model name (for LoRA)')
    parser.add_argument('--use_lora', action='store_true', default=True,
                        help='Whether model is LoRA adapter')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for generation')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--lang', type=str, default='zh', choices=['zh', 'en'],
                        help='Language code')
    parser.add_argument('--bertscore_model', type=str, default=None,
                        help='Specific BERTScore model')
    parser.add_argument('--no_save_predictions', action='store_true',
                        help='Do not save predictions')

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        test_jsonl_path=args.test_data,
        output_path=args.output,
        base_model=args.base_model,
        use_lora=args.use_lora,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lang=args.lang,
        bertscore_model=args.bertscore_model,
        save_predictions=not args.no_save_predictions
    )
