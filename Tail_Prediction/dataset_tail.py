import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict

def load_triplets(file_path):
    """Load triplets from a TSV file."""
    triplets_df = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    return triplets_df

def get_one_hop_head_entity_neighbors(entity, triplets, max_degree=15):
    """
    Extracts one-hop relations and connected entities for a given head entity,
    but only keeps the top `max_degree` entities based on their degree.
    """
    # Calculate degrees for all entities
    entity_degrees = defaultdict(int)
    for head, relation, tail in triplets.values:
        entity_degrees[head] += 1
        entity_degrees[tail] += 1

    # Get all one-hop relations and connected entities for the head entity
    one_hop_neighbors = triplets[triplets['head'] == entity][['relation', 'tail']].values.tolist()

    # Sort the neighbors by degree (descending) and take the top `max_degree`
    sorted_neighbors = sorted(one_hop_neighbors, key=lambda x: entity_degrees[x[1]], reverse=True)
    top_neighbors = sorted_neighbors[:max_degree]

    # Format the result as "head-relation-tail"
    result = [f"{entity}-{rel}-{tail}" for rel, tail in top_neighbors]
    return result

def get_relation_neighbors(relation, triplets, max_degree=5):
    """
    Extracts one-hop relations and connected entities for a given relation,
    but only keeps the top `max_degree` entities based on their degree.
    """
    # Calculate degrees for all entities
    entity_degrees = defaultdict(int)
    for head, relation, tail in triplets.values:
        entity_degrees[head] += 1
        entity_degrees[tail] += 1

    # Get all one-hop relations and connected entities for the given relation
    one_hop_neighbors = triplets[triplets['relation'] == relation][['head', 'tail']].values.tolist()

    # Sort the neighbors by degree (descending) and take the top `max_degree`
    sorted_neighbors = sorted(one_hop_neighbors, key=lambda x: entity_degrees[x[1]], reverse=True)
    top_neighbors = sorted_neighbors[:max_degree]

    # Format the result as "head-relation-tail"
    result = [f"{head}-{relation}-{tail}" for head, tail in top_neighbors]
    return result

class KGDataset(Dataset):
    def __init__(self, triplets, tokenizer, entity_to_idx, all_triplets,
                 max_degree_head=15, max_degree_relation=5):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.entity_to_idx = entity_to_idx
        self.all_triplets = all_triplets
        self.max_degree_head = max_degree_head
        self.max_degree_relation = max_degree_relation

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        row = self.triplets.iloc[idx]
        head, relation, tail = row['head'], row['relation'], row['tail']

        head_neighbors = get_one_hop_head_entity_neighbors(head, self.all_triplets, self.max_degree_head)
        relation_neighbors = get_relation_neighbors(relation, self.all_triplets, self.max_degree_relation)

        head_neighbors_str = " ".join(head_neighbors)
        relation_neighbors_str = " ".join(relation_neighbors)

        inputs = self.tokenizer(
            f"{head} [SEP] {head_neighbors_str} [SEP] {relation} [SEP] {relation_neighbors_str}",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # Remove batch dimension
        label = torch.tensor(self.entity_to_idx[tail])
        return inputs, label