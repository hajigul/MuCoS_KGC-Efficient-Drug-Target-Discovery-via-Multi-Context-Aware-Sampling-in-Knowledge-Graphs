import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict



# ==================== LOAD TRIPLETS FUNCTION ====================
# This function reads a knowledge graph file where each line contains
# a triplet (head, relation, tail) separated by tabs.
# It converts the file into a pandas DataFrame so that the data
# can be easily accessed and processed later in the pipeline.

def load_triplets(file_path):
    """Load triplets from a tab-separated file."""
    triplets_df = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    return triplets_df



# ==================== DATASET CLASS FOR RELATION PREDICTION ====================
# This custom Dataset class prepares data for training a transformer model.
# It converts each triplet into a text format and adds graph-based context
# (neighbor information) for both head and tail entities.

class KGRelationDataset(Dataset):
    """Dataset for relation prediction."""
    
    def __init__(self, triplets, tokenizer, relation_to_idx, entity_incoming_neighbors,
                 max_length=128):
        """
        Args:
            triplets: DataFrame of triplets (head, relation, tail)
            tokenizer: HuggingFace tokenizer
            relation_to_idx: dict mapping relation to label index
            entity_incoming_neighbors: precomputed dict of neighbor strings per entity
            max_length: max token length
        """
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.relation_to_idx = relation_to_idx
        self.entity_incoming_neighbors = entity_incoming_neighbors
        self.max_length = max_length

    def __len__(self):
        return len(self.triplets)
    


    # This method defines how a single data sample is created.
    # It combines entity names and their graph context into one text input,
    # tokenizes it, and returns it along with the relation label.

    def __getitem__(self, idx):
        row = self.triplets.iloc[idx]
        head, relation, tail = row['head'], row['relation'], row['tail']
        
        # Get precomputed neighbor info for head and tail
        head_context = self.entity_incoming_neighbors.get(head, [])
        tail_context = self.entity_incoming_neighbors.get(tail, [])
        
        head_context_str = " ".join(head_context)
        tail_context_str = " ".join(tail_context)
        
        # Input text: head, head context, tail, tail context
        text = f"{head} [SEP] {head_context_str} [SEP] {tail} [SEP] {tail_context_str}"
        
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length",
                                truncation=True, max_length=self.max_length)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        
        label = torch.tensor(self.relation_to_idx[relation])
        return inputs, label


# ==================== PRECOMPUTE GRAPH CONTEXT ====================
# This function prepares structural information for each entity before training.
# It computes:
#   1. Degree of each entity (how connected it is)
#   2. Incoming neighbors (which entities point to it)
#   3. Filtered neighbor context (removes very high-degree nodes to reduce noise)
#   4. Second-level context (neighbors of neighbors) to enrich representation
# This preprocessing step improves both efficiency and model performance.

def precompute_entity_info(all_triplets, max_degree):
    """
    Precompute for each entity a string of filtered incoming neighbors.
    """
    # Step 1: compute degrees for all entities
    entity_degrees = defaultdict(int)
    for head, relation, tail in all_triplets.values:
        entity_degrees[head] += 1
        entity_degrees[tail] += 1
    
    # Step 2: build incoming relations map: tail -> list of (relation, head)
    incoming_map = defaultdict(list)
    for head, relation, tail in all_triplets.values:
        incoming_map[tail].append((relation, head))
    
    # Step 3: for each entity, build its filtered incoming neighbor context
    entity_incoming_neighbors = {}
    for entity in entity_degrees.keys():
        neighbors = []
        for rel, source in incoming_map.get(entity, []):
            if entity_degrees[source] > max_degree:
                continue
            # For the source node, also get its incoming neighbors (filtered)
            incoming_of_source = []
            for rel_in, head_in in incoming_map.get(source, []):
                if entity_degrees[head_in] <= max_degree:
                    incoming_of_source.append(f"{rel_in} {head_in}")
            incoming_str = " ".join(incoming_of_source)
            neighbors.append(f"{rel} {source} [IN] {incoming_str}")
        entity_incoming_neighbors[entity] = neighbors
    return entity_degrees, entity_incoming_neighbors



# ==================== CREATE DATALOADERS ====================
# This function creates PyTorch DataLoaders for training, validation, and testing.
# It first converts raw triplets into dataset objects, then wraps them into
# DataLoaders which handle batching, shuffling, and efficient iteration.

def create_dataloaders(train_triplets, valid_triplets, test_triplets, tokenizer,
                       relation_to_idx, entity_incoming_neighbors,
                       batch_size=16, max_length=128):
    """Create DataLoaders for train, validation, and test sets."""
    
    train_dataset = KGRelationDataset(train_triplets, tokenizer, relation_to_idx,
                                      entity_incoming_neighbors, max_length)
    valid_dataset = KGRelationDataset(valid_triplets, tokenizer, relation_to_idx,
                                      entity_incoming_neighbors, max_length)
    test_dataset = KGRelationDataset(test_triplets, tokenizer, relation_to_idx,
                                     entity_incoming_neighbors, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader