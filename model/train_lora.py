import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb


class MedicalQADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        input_text = f"medical question: {item['human']}"
        target_text = item['assistant']

        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True
        )

        labels = self.tokenizer(
            target_text,
            max_length=self.max_length,
            truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

def train_lora(
    train_jsonl_path,
    model_name='google/mt5-base',
    output_dir='model/lora_checkpoints',
    batch_size=8,
    epochs=10,
    learning_rate=3e-4,
    max_length=512,
    lora_r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    warmup_ratio=0.1,
    weight_decay=0.01,
    use_wandb=True,
    wandb_project='medical-qa-lora',
    wandb_run_name=None,
    val_split=0.1
):
    # init wandb
    if use_wandb:
        dataset_name = os.path.basename(os.path.dirname(train_jsonl_path))
        run_name = wandb_run_name or f"{dataset_name}_lora_e{epochs}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                'model': model_name,
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'lora_r': lora_r,
                'lora_alpha': lora_alpha,
                'lora_dropout': lora_dropout,
                'warmup_ratio': warmup_ratio,
                'weight_decay': weight_decay,
                'max_length': max_length,
                'dataset': dataset_name
            }
        )

    tokenizer = MT5Tokenizer.from_pretrained(model_name)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    model = MT5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype
    )

    if torch.cuda.is_available():
        model = model.cuda()

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q", "v"],
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    full_dataset = MedicalQADataset(train_jsonl_path, tokenizer, max_length)

    # split into train/val
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=50,
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps",
        bf16=use_bf16,
        fp16=False,
        gradient_accumulation_steps=2,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        label_names=["labels"],
        lr_scheduler_type="cosine",
        weight_decay=weight_decay,
        max_grad_norm=1.0,
        logging_first_step=True,
        optim="adamw_torch",
        report_to="wandb" if use_wandb else "none",
        run_name=wandb_run_name,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    final_dir = os.path.join(output_dir, 'final')
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    if use_wandb:
        wandb.finish()

    print(f"lora training complete - saved to {final_dir}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python train_lora.py <jsonl_path> <output_dir>")
        sys.exit(1)

    train_jsonl_path = sys.argv[1]
    output_dir = sys.argv[2]

    train_lora(train_jsonl_path, output_dir=output_dir)
