import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
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

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels
        }

def train_epoch(model, dataloader, optimizer, scheduler, device, use_wandb):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if use_wandb:
            wandb.log({'batch_loss': loss.item(), 'learning_rate': scheduler.get_last_lr()[0]})

    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(dataloader)

def train(
    train_jsonl_path,
    model_name='google/mt5-base',
    output_dir='model/checkpoints',
    batch_size=4,
    epochs=10,
    learning_rate=5e-5,
    max_length=512,
    use_wandb=True,
    wandb_project='medical-qa-full',
    wandb_run_name=None,
    val_split=0.1
):
    # init wandb
    if use_wandb:
        dataset_name = os.path.basename(os.path.dirname(train_jsonl_path))
        run_name = wandb_run_name or f"{dataset_name}_full_e{epochs}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                'model': model_name,
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'max_length': max_length,
                'dataset': dataset_name
            }
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    full_dataset = MedicalQADataset(train_jsonl_path, tokenizer, max_length)

    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device, use_wandb)
        val_loss = eval_epoch(model, val_dataloader, device)

        print(f"epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

        epoch_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)

    final_dir = os.path.join(output_dir, 'final')
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    if use_wandb:
        wandb.finish()

    print(f"training complete - saved to {output_dir}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python train_full.py <jsonl_path> <output_dir>")
        sys.exit(1)

    train_jsonl_path = sys.argv[1]
    output_dir = sys.argv[2]

    train(train_jsonl_path, output_dir=output_dir)
