import os
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import warnings

import config
from data_loader import load_triplets, precompute_entity_info, KGRelationDataset
from model import get_model_and_tokenizer
from utils import evaluate_relation_model, save_test_results, save_checkpoint

warnings.filterwarnings("ignore")



# ==================== DDP SETUP AND CLEANUP ====================
# These functions set up the distributed training environment for PyTorch.
# - `setup(rank, world_size)`: Initializes process groups, sets GPU device for this rank,
#   and enables cuDNN benchmark for better performance.
# - `cleanup()`: Destroys the process group after training to release resources.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.MASTER_PORT
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True


def cleanup():
    dist.destroy_process_group()



# ==================== TRAIN FUNCTION ====================
# This is the main training loop executed by each process in DDP.
# It handles:
# - Loading dataset triplets
# - Precomputing graph context for entities (neighbors)
# - Initializing the model and wrapping it with DistributedDataParallel (DDP)
# - Creating DataLoaders with DistributedSampler for proper sharding across GPUs
# - Gradient accumulation for memory efficiency
# - Epoch-wise training, logging, evaluation, and checkpointing

def train(rank, world_size, dataset_config):
    """DDP training function – receives the current dataset config"""
    setup(rank, world_size)

    if rank == 0:
        print("=" * 70)
        print(f"Knowledge Graph Completion - Relation Prediction")
        print(f"DATASET: {dataset_config['name']}")
        print("=" * 70)

    # Load triplets using the passed dataset config
    train_triplets = load_triplets(dataset_config["TRAIN_FILE_PATH"])
    valid_triplets = load_triplets(dataset_config["VALID_FILE_PATH"])
    test_triplets = load_triplets(dataset_config["TEST_FILE_PATH"])

    # Build relation mapping
    all_relations = pd.concat([train_triplets['relation'],
                               valid_triplets['relation'],
                               test_triplets['relation']]).unique().tolist()
    relation_to_idx = {rel: idx for idx, rel in enumerate(all_relations)}
    num_relations = len(relation_to_idx)

    all_triplets = pd.concat([train_triplets, valid_triplets, test_triplets])



        # -------------------- PRECOMPUTE ENTITY INFO --------------------
    # Precomputes neighbor information for all entities and shares it
    # with other processes via temporary file to avoid redundant computation.

    if rank == 0:
        print("Precomputing entity information...")
        entity_degrees, entity_incoming_neighbors = precompute_entity_info(all_triplets, config.MAX_DEGREE)
        torch.save(entity_incoming_neighbors, '/tmp/entity_neighbors.pt')
        print("Precomputation complete. Saved to /tmp/")

    dist.barrier()

    if rank != 0:
        entity_incoming_neighbors = torch.load('/tmp/entity_neighbors.pt')

    dist.barrier()

    if rank == 0:
        print("All ranks have loaded precomputed data.")



        # -------------------- MODEL INITIALIZATION --------------------
    # Load the selected transformer model and tokenizer, move model to GPU,
    # and wrap it in DistributedDataParallel for multi-GPU training.

    model, tokenizer = get_model_and_tokenizer(config.MODEL_NAME, num_labels=num_relations)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)



        # -------------------- DATALOADER SETUP --------------------
    # Training dataset uses DistributedSampler for proper data sharding.
    # Validation and test datasets use standard DataLoader.

    train_dataset = KGRelationDataset(train_triplets, tokenizer, relation_to_idx,
                                      entity_incoming_neighbors, max_length=config.MAX_LENGTH)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.PER_GPU_BATCH_SIZE,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    # Validation / Test dataloaders
    valid_dataset = KGRelationDataset(valid_triplets, tokenizer, relation_to_idx,
                                      entity_incoming_neighbors, max_length=config.MAX_LENGTH)
    test_dataset = KGRelationDataset(test_triplets, tokenizer, relation_to_idx,
                                     entity_incoming_neighbors, max_length=config.MAX_LENGTH)

    valid_dataloader = DataLoader(valid_dataset, batch_size=config.PER_GPU_BATCH_SIZE,
                                  shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.PER_GPU_BATCH_SIZE,
                                 shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    gradient_accumulation_steps = 2

    if rank == 0:
        print(f"Starting training for {config.NUM_EPOCHS} epochs on {dataset_config['name']}...")



        # -------------------- TRAINING LOOP --------------------
    # For each epoch, iterate over batches, compute loss, and update gradients.
    # Gradient accumulation is used to simulate larger batch sizes.

    for epoch in range(config.NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        total_batches = len(train_dataloader)

        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs = {k: v.to(rank, non_blocking=True) for k, v in inputs.items()}
            labels = labels.to(rank, non_blocking=True)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * gradient_accumulation_steps

            if rank == 0 and (batch_idx + 1) % 100 == 0:
                current_loss = train_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{total_batches} | Loss: {loss.item() * gradient_accumulation_steps:.4f} | Avg Loss: {current_loss:.4f}")

        if rank == 0:
            avg_train_loss = train_loss / total_batches
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Avg Train Loss: {avg_train_loss:.4f}")
            print(f"{'='*50}")

            print("Evaluating on test set...")
            test_results = evaluate_relation_model(
                model.module, test_dataloader, rank,
                relation_to_idx, test_triplets, tokenizer,
                entity_incoming_neighbors, max_length=config.MAX_LENGTH
            )

            # Print metrics including MR
            print(f"  MR: {test_results['MR']:.2f}")
            print(f"  MRR: {test_results['MRR']:.4f}")
            print(f"  Hits@1: {test_results['Hits@1']:.4f}")
            print(f"  Hits@3: {test_results['Hits@3']:.4f}")
            print(f"  Hits@5: {test_results['Hits@5']:.4f}")
            print(f"  Hits@10: {test_results['Hits@10']:.4f}\n")

            save_test_results(epoch, test_results, dataset_config["MODEL_SAVE_PATH"], task='relation')
            save_checkpoint(model.module, optimizer, epoch + 1,
                           os.path.join(dataset_config["MODEL_SAVE_PATH"], f'checkpoint_epoch_{epoch+1}.pth'))

    if rank == 0:
        print("\nTraining Completed Successfully for this dataset!")
        model.module.save_pretrained(dataset_config["MODEL_SAVE_PATH"])
        tokenizer.save_pretrained(dataset_config["MODEL_SAVE_PATH"])
        if os.path.exists('/tmp/entity_neighbors.pt'):
            os.remove('/tmp/entity_neighbors.pt')

    cleanup()



# ==================== MAIN SCRIPT ====================
# This block loops through all datasets sequentially.
# For each dataset:
# - prints dataset info
# - spawns multiple processes (one per GPU) for DDP training

if __name__ == '__main__':
    print("=" * 100)
    print("KNOWLEDGE GRAPH COMPLETION – SEQUENTIAL MULTI-DATASET TRAINING")
    print("=" * 100)
    print(f"Total datasets: {len(config.DATASETS)}\n")

    for idx, dataset in enumerate(config.DATASETS):
        print(f"\n{'='*100}")
        print(f"DATASET {idx+1}/{len(config.DATASETS)} → {dataset['name'].upper()}")
        print(f"Model will be saved to: {dataset['MODEL_SAVE_PATH']}")
        print(f"{'='*100}\n")

        world_size = config.NUM_GPUS
        # Pass the full dataset config to every spawned process
        mp.spawn(train, args=(world_size, dataset), nprocs=world_size, join=True)

    print("\n ALL DATASETS PROCESSED SUCCESSFULLY! ")