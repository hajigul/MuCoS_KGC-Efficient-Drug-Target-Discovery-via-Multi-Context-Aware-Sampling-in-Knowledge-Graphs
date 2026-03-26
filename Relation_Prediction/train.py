import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW

from dataset import KGDataset
from utils import save_test_results


def evaluate_model(model, dataloader, device, relation_to_idx, test_triplets, tokenizer):
    model.eval()
    rankings = []
    labels = []

    with torch.no_grad():
        for idx, _ in enumerate(dataloader):          # we only need the index order
            row = test_triplets.iloc[idx]
            head, true_relation, tail = row['head'], row['relation'], row['tail']

            inputs = tokenizer(
                f"{head} [SEP] {tail}",
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128
            )
            inputs = {k: v.squeeze(0).unsqueeze(0).to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits.squeeze().cpu().numpy()
            scores = np.exp(logits) / np.sum(np.exp(logits))

            rankings.append(scores)
            labels.append(relation_to_idx[true_relation])

    rankings = np.array(rankings)
    labels = np.array(labels)

    mrr = np.mean([1.0 / (np.argsort(scores)[::-1].tolist().index(label) + 1)
                   for scores, label in zip(rankings, labels)])
    hits_at_1 = np.mean([label in np.argsort(scores)[::-1][:1] for scores, label in zip(rankings, labels)])
    hits_at_3 = np.mean([label in np.argsort(scores)[::-1][:3] for scores, label in zip(rankings, labels)])
    hits_at_5 = np.mean([label in np.argsort(scores)[::-1][:5] for scores, label in zip(rankings, labels)])
    hits_at_10 = np.mean([label in np.argsort(scores)[::-1][:10] for scores, label in zip(rankings, labels)])

    return {
        'MRR': mrr,
        'Hits@1': hits_at_1,
        'Hits@3': hits_at_3,
        'Hits@5': hits_at_5,
        'Hits@10': hits_at_10
    }


def train_and_evaluate(model_name, tokenizer_class, model_class, save_path, num_epochs,
                       train_triplets, valid_triplets, test_triplets,
                       relation_to_idx, all_triplets, device):
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, num_labels=len(relation_to_idx))
    model.to(device)

    # Datasets & DataLoaders
    train_dataset = KGDataset(train_triplets, tokenizer, relation_to_idx, all_triplets)
    valid_dataset = KGDataset(valid_triplets, tokenizer, relation_to_idx, all_triplets)
    test_dataset = KGDataset(test_triplets, tokenizer, relation_to_idx, all_triplets)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_dataloader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                outputs = model(**inputs, labels=labels)
                valid_loss += outputs.loss.item()
        avg_valid_loss = valid_loss / len(valid_dataloader)

        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}")

        # Test evaluation & save results
        test_results = evaluate_model(model, test_dataloader, device, relation_to_idx,
                                      test_triplets, tokenizer)
        save_test_results(epoch + 1, test_results, save_path)

    # Save final model
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to: {save_path}")