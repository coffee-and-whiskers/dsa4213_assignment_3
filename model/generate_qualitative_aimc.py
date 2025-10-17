#!/usr/bin/env python3

import os
import json
import torch
from tqdm import tqdm
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from peft import PeftModel
import argparse


def load_questions(json_path):
    """Load questions from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['questions']


def load_model_and_tokenizer(model_path, base_model='google/mt5-base', use_lora=True):
    """Load model and tokenizer."""
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


def generate_with_temperature(model, tokenizer, questions, temperature, max_length=512):
    """Generate answers with specific temperature."""
    results = []

    for question in tqdm(questions, desc=f"Generating (temp={temperature})"):
        input_text = f"medical question: {question}"
        encoded = tokenizer(
            input_text,
            max_length=max_length,
            truncation=True,
            return_tensors='pt'
        )

        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_length=max_length,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                top_p=0.9,
                top_k=50,
                no_repeat_ngram_size=3,
                num_return_sequences=1
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            'question': question,
            'answer': answer,
            'temperature': temperature
        })

    return results


def generate_qualitative_outputs(
    model_configs,
    questions_path,
    output_dir,
    temperatures=[0.7, 1.0, 1.3],
    max_length=512,
    lang='en'
):
    """Generate qualitative outputs for all models and temperatures."""

    # Load questions
    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions from {questions_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for model_name, model_config in model_configs.items():
        print(f"\n{'='*80}")
        print(f"Processing model: {model_name}")
        print(f"{'='*80}")

        # Load model
        model, tokenizer = load_model_and_tokenizer(
            model_path=model_config['path'],
            base_model=model_config.get('base_model', 'google/mt5-base'),
            use_lora=model_config.get('use_lora', True)
        )

        model_results = {}

        # Generate for each temperature
        for temp in temperatures:
            print(f"\nGenerating with temperature={temp}")
            temp_results = generate_with_temperature(
                model, tokenizer, questions, temp, max_length
            )
            model_results[f"temp_{temp}"] = temp_results

        all_results[model_name] = model_results

        # Free memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Save all results
    output_path = os.path.join(output_dir, f'qualitative_generation_{lang}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"All results saved to: {output_path}")
    print(f"{'='*80}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate qualitative outputs for AIMC models')
    parser.add_argument('--questions_path', type=str, required=True,
                        help='Path to questions JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save outputs')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.7, 1.0, 1.3],
                        help='List of temperatures to use')
    parser.add_argument('--cuda_device', type=int, default=None,
                        help='CUDA device to use (will auto-select if not specified)')

    args = parser.parse_args()

    # Set CUDA device if specified
    if args.cuda_device is not None:
        torch.cuda.set_device(args.cuda_device)

    # Define model configurations for AIMC
    model_configs = {
        'full': {
            'path': 'model/checkpoints/aimc_full/final',
            'use_lora': False
        },
        'baseline_lora': {
            'path': 'model/checkpoints/aimc_lora_baseline/final',
            'base_model': 'google/mt5-base',
            'use_lora': True
        },
        'gs_1': {
            'path': 'model/checkpoints/aimc_lora_gs/aimc_gs_1_lr2.0e-04_r16_alpha64_do0.1/final',
            'base_model': 'google/mt5-base',
            'use_lora': True
        },
        'gs_2': {
            'path': 'model/checkpoints/aimc_lora_gs/aimc_gs_2_lr1.5e-04_r12_alpha48_do0.1/final',
            'base_model': 'google/mt5-base',
            'use_lora': True
        },
        'gs_3': {
            'path': 'model/checkpoints/aimc_lora_gs/aimc_gs_3_lr2.0e-04_r12_alpha48_do0.05/final',
            'base_model': 'google/mt5-base',
            'use_lora': True
        },
        'gs_4': {
            'path': 'model/checkpoints/aimc_lora_gs/aimc_gs_4_lr2.0e-04_r12_alpha48_do0.15/final',
            'base_model': 'google/mt5-base',
            'use_lora': True
        }
    }

    # Generate qualitative outputs
    generate_qualitative_outputs(
        model_configs=model_configs,
        questions_path=args.questions_path,
        output_dir=args.output_dir,
        temperatures=args.temperatures,
        max_length=args.max_length,
        lang='en'
    )
