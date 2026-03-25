import torch
from torch.utils.data import Dataset

from utils import get_one_hop_head_entity_neighbors, get_one_hop_tail_entity_neighbors


class KGDataset(Dataset):
    def __init__(self, triplets, tokenizer, relation_to_idx, all_triplets):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.relation_to_idx = relation_to_idx
        self.all_triplets = all_triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        row = self.triplets.iloc[idx]
        head, relation, tail = row['head'], row['relation'], row['tail']

        head_neighbors = get_one_hop_head_entity_neighbors(head, self.all_triplets, max_degree=20)
        tail_neighbors = get_one_hop_tail_entity_neighbors(tail, self.all_triplets, max_degree=20)

        head_neighbors_str = " ".join(head_neighbors)
        tail_neighbors_str = " ".join(tail_neighbors)

        inputs = self.tokenizer(
            f"{head} [SEP] {head_neighbors_str} [SEP] {tail} [SEP] {tail_neighbors_str}",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        label = torch.tensor(self.relation_to_idx[relation])
        return inputs, label