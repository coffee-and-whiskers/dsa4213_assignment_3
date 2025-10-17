#!/usr/bin/env python3

import os
import re
import json
import pandas as pd
from pathlib import Path


def parse_log_file(log_path):
    """Parse evaluation log file to extract metrics."""
    metrics = {}

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Define patterns for each metric
    patterns = {
        'rouge1': r'rouge1:\s+([\d.]+)',
        'rouge2': r'rouge2:\s+([\d.]+)',
        'rougeL': r'rougeL:\s+([\d.]+)',
        'f1': r'f1:\s+([\d.]+)',
        'bertscore_f1': r'bertscore_f1:\s+([\d.]+)',
        'exact_match': r'exact_match:\s+([\d.]+)'
    }

    # Extract metrics using regex
    for metric_name, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[metric_name] = float(match.group(1))
        else:
            metrics[metric_name] = None

    return metrics


def collate_all_metrics(logs_dir, output_path):
    """Collate all metrics from evaluation logs."""

    logs_path = Path(logs_dir)

    if not logs_path.exists():
        print(f"Error: Directory {logs_dir} does not exist.")
        return

    # Get all log files
    log_files = list(logs_path.glob('eval_*.log'))

    if not log_files:
        print(f"No evaluation log files found in {logs_dir}")
        return

    print(f"Found {len(log_files)} evaluation log files")

    results = []

    for log_file in sorted(log_files):
        print(f"Processing {log_file.name}...")

        # Extract model info from filename
        filename = log_file.stem  # Remove .log extension

        # Determine dataset and model type
        if 'aimc' in filename:
            dataset = 'aimc'
        elif 'huatuo' in filename:
            dataset = 'huatuo'
        else:
            dataset = 'unknown'

        # Determine model type
        if 'full' in filename:
            model_type = 'full'
        elif 'baseline' in filename:
            model_type = 'baseline_lora'
        elif 'gs_1' in filename:
            model_type = 'gs_1'
        elif 'gs_2' in filename:
            model_type = 'gs_2'
        elif 'gs_3' in filename:
            model_type = 'gs_3'
        elif 'gs_4' in filename:
            model_type = 'gs_4'
        else:
            model_type = 'unknown'

        # Parse metrics
        metrics = parse_log_file(log_file)

        # Combine all info
        result = {
            'dataset': dataset,
            'model_type': model_type,
            'log_file': log_file.name,
            **metrics
        }

        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by dataset and model type
    df = df.sort_values(['dataset', 'model_type'])

    # Save to CSV
    csv_path = output_path.replace('.json', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to CSV: {csv_path}")

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to JSON: {output_path}")

    # Print summary table
    print("\n" + "="*100)
    print("EVALUATION METRICS SUMMARY")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    # Print best models per dataset
    print("\nBEST MODELS BY METRIC:")
    print("-"*100)

    for dataset in df['dataset'].unique():
        if dataset == 'unknown':
            continue
        print(f"\n{dataset.upper()} Dataset:")
        dataset_df = df[df['dataset'] == dataset]

        for metric in ['rouge1', 'rouge2', 'rougeL', 'f1', 'bertscore_f1', 'exact_match']:
            if metric in dataset_df.columns:
                best_idx = dataset_df[metric].idxmax()
                best_model = dataset_df.loc[best_idx, 'model_type']
                best_score = dataset_df.loc[best_idx, metric]
                print(f"  {metric:20s}: {best_model:15s} ({best_score:.4f})")

    print("\n" + "="*100)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Collate evaluation metrics from log files')
    parser.add_argument('--logs_dir', type=str, default='logs/eval',
                        help='Directory containing evaluation log files')
    parser.add_argument('--output', type=str, default='results/collated_metrics.json',
                        help='Output path for collated metrics (JSON)')

    args = parser.parse_args()

    # Collate metrics
    collate_all_metrics(args.logs_dir, args.output)
