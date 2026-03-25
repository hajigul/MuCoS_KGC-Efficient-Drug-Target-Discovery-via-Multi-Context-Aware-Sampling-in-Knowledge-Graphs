import os
import pandas as pd
from collections import defaultdict


def load_triplets(file_path):
    triplets_df = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    return triplets_df


def get_one_hop_head_entity_neighbors(entity, triplets, max_degree=20):
    """Extracts one-hop neighbors for a head entity (top `max_degree` by degree)."""
    entity_degrees = defaultdict(int)
    for head, relation, tail in triplets.values:
        entity_degrees[head] += 1
        entity_degrees[tail] += 1

    one_hop_neighbors = triplets[triplets['head'] == entity][['relation', 'tail']].values.tolist()
    sorted_neighbors = sorted(one_hop_neighbors, key=lambda x: entity_degrees[x[1]], reverse=True)
    top_neighbors = sorted_neighbors[:max_degree]

    return [f"{entity}-{rel}-{tail}" for rel, tail in top_neighbors]


def get_one_hop_tail_entity_neighbors(entity, triplets, max_degree=20):
    """Extracts one-hop neighbors for a tail entity (top `max_degree` by degree)."""
    entity_degrees = defaultdict(int)
    for head, relation, tail in triplets.values:
        entity_degrees[head] += 1
        entity_degrees[tail] += 1

    one_hop_neighbors = triplets[triplets['tail'] == entity][['head', 'relation']].values.tolist()
    sorted_neighbors = sorted(one_hop_neighbors, key=lambda x: entity_degrees[x[0]], reverse=True)
    top_neighbors = sorted_neighbors[:max_degree]

    return [f"{head}-{rel}-{entity}" for head, rel in top_neighbors]


def save_test_results(epoch, test_results, save_path):
    """Appends test metrics (MRR, Hits@K) to a results file."""
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "test_results.txt")

    epoch_results = (f"{test_results['MRR']:.4f}\t"
                     f"{test_results['Hits@1']:.4f}\t"
                     f"{test_results['Hits@3']:.4f}\t"
                     f"{test_results['Hits@5']:.4f}\t"
                     f"{test_results['Hits@10']:.4f}\n")

    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("MRR\tHit@1\tHit@3\tHit@5\tHit@10\n")

    with open(file_path, 'a') as f:
        f.write(epoch_results)