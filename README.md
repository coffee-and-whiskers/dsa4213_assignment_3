# Medical QA Fine-tuning with MT5

Fine-tuning multilingual T5 (MT5-base) on bilingual medical question-answering datasets using full fine-tuning and LoRA (Low-Rank Adaptation).

## Project Structure

```
llm/
├── data/                    # Training and test datasets (AIMC English, Huatuo Chinese)
├── model/                   # Training scripts, evaluation scripts, and model checkpoints
│   ├── checkpoints/        # Trained model weights (full + LoRA + grid search variants)
│   └── scripts/            # Training orchestration and evaluation scripts
├── results/                 # Evaluation metrics and predictions (JSON/JSONL)
├── logs/                    # Training and evaluation logs
├── wandb/                   # Weights & Biases experiment tracking
├── requirements.txt         # Python dependencies
└── dockerfile              # Docker environment setup
```

## Installation

### Using Docker (Recommended)

```bash
docker build -t medical-qa-finetuning .
docker run --gpus all -it -v $(pwd):/workspace medical-qa-finetuning
```

### Using Conda/Pip

```bash
conda create -n llm python=3.10
conda activate llm
pip install -r requirements.txt
```

## Datasets

- **AIMC** (English Medical QA): `data/aimc_formatted/`
- **Huatuo** (Chinese Medical QA): `data/huatuo_formatted/`

Each dataset contains train/test splits in JSONL format with `human` (question) and `assistant` (answer) fields.

## Training

All training jobs should be run using the scripts in `model/scripts/`:

### Run Preliminary Training (Full + Baseline LoRA)
```bash
bash model/scripts/run_training_prelim.sh
```

### Run Grid Search Training
```bash
python model/scripts/train_lora_aimc_gs.py
python model/scripts/train_lora_huatuo_gs.py
```

## Evaluation

All evaluation should be run using the scripts in `model/scripts/`:

### Evaluate Baseline Models
```bash
bash model/scripts/run_eval_baseline_lora.sh
```

### Evaluate Full Fine-tuned Models
```bash
bash model/scripts/run_eval_full.sh
```

### Evaluate Grid Search Models
```bash
bash model/scripts/run_eval_gs_loras.sh
```

### Collate Results
```bash
python model/collate_metrics.py
```

Results will be saved to `results/collated_metrics.json` and `results/collated_metrics.csv`.

### Generate Qualitative Samples
```bash
bash model/scripts/run_qualitative_generation.sh
```

## Evaluation Metrics

- **ROUGE-1, ROUGE-2, ROUGE-L**: N-gram overlap
- **Token F1**: Token-level precision/recall
- **BERTScore F1**: Semantic similarity (bert-base-chinese for Chinese, bert-base-uncased for English)
- **Exact Match**: Exact string match percentage

## Model Configurations

**Base Model**: `google/mt5-base` (multilingual T5)

**Training Variants**:
- Full fine-tuning (10 epochs, lr=5e-5, batch_size=4)
- LoRA baseline (10 epochs, lr=3e-4, r=8, alpha=32)
- LoRA grid search (8 configurations with varying lr, r, alpha, dropout)

## Requirements

- Python 3.10+
- CUDA-capable GPU (12GB+ VRAM recommended)
- 60GB disk space for datasets and checkpoints

## Repository

GitHub: [Add your repository URL here]

**Note**: Model checkpoints (48GB) are not included in the repository. Please download them separately or train from scratch using the provided scripts.
