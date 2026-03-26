import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW
from dataset_tail import KGDataset, get_one_hop_head_entity_neighbors
from utils_tail import save_test_results, save_checkpoint, load_checkpoint

def evaluate_model(model, dataloader, device, entity_to_idx, test_triplets, tokenizer):
    """Evaluate the model on test set and compute metrics."""
    model.eval()
    rankings = []
    labels = []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            row = test_triplets.iloc[idx]
            head, relation, true_tail = row['head'], row['relation'], row['tail']

            head_neighbors = get_one_hop_head_entity_neighbors(head, test_triplets)  # Note: using test_triplets for neighbor extraction
            head_neighbors_str = " ".join(head_neighbors)

            inputs = tokenizer(
                f"{head} [SEP] {head_neighbors_str} [SEP] {relation}",
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128
            )
            inputs = {key: val.squeeze(0).unsqueeze(0).to(device) for key, val in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits.squeeze().cpu().numpy()
            scores = np.exp(logits) / np.sum(np.exp(logits))  # softmax to get probabilities
            rankings.append(scores)
            labels.append(entity_to_idx[true_tail])

    rankings = np.array(rankings)
    labels = np.array(labels)

    # Calculate MRR and Hits@k
    mrr = np.mean([1.0 / (np.argsort(scores)[::-1].tolist().index(label) + 1) for scores, label in zip(rankings, labels)])
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

def train_and_evaluate(
    model_name,
    tokenizer_class,
    model_class,
    save_path,
    num_epochs,
    train_triplets,
    valid_triplets,
    test_triplets,
    all_triplets,
    entity_to_idx,
    batch_size=16,
    learning_rate=5e-5,
    max_length=128,
    max_degree_head=15,
    max_degree_relation=5,
    device='cpu'
):
    """Train and evaluate the model."""
    # Load tokenizer and model
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, num_labels=len(entity_to_idx))
    model.to(device)

    # Prepare Datasets and DataLoaders
    train_dataset = KGDataset(train_triplets, tokenizer, entity_to_idx, all_triplets,
                              max_degree_head, max_degree_relation)
    valid_dataset = KGDataset(valid_triplets, tokenizer, entity_to_idx, all_triplets,
                              max_degree_head, max_degree_relation)
    test_dataset = KGDataset(test_triplets, tokenizer, entity_to_idx, all_triplets,
                             max_degree_head, max_degree_relation)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Load checkpoint if exists
    checkpoint_path = os.path.join(save_path, 'checkpoint.pth')
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    print(f"Resuming from epoch {start_epoch + 1}" if start_epoch > 0 else "Starting from scratch")

    model.train()

    for epoch in range(start_epoch, num_epochs):
        train_loss = 0.0
        for batch in train_dataloader:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
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
            for batch in valid_dataloader:
                inputs, labels = batch
                inputs = {key: val.to(device) for key, val in inputs.items()}
                labels = labels.to(device)

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                valid_loss += loss.item()
        avg_valid_loss = valid_loss / len(valid_dataloader)

        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}")

        # Evaluate on test set
        test_results = evaluate_model(model, test_dataloader, device, entity_to_idx, test_triplets, tokenizer)
        save_test_results(epoch + 1, test_results, save_path)

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

    # Save final model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)