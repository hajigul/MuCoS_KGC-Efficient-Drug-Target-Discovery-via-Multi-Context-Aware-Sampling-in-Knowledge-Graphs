import os
import pandas as pd
from collections import defaultdict


def load_triplets(file_path):
    """Load triplets and filter only DRUG_TARGET_GENE and DRUG_TARGET_PATHWAY relations."""
    triplets_df = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    
    # Filter for specific drug-target relations (as in your original code)
    drug_target_triplets = triplets_df[
        (triplets_df['relation'] == 'DRUG_TARGET_GENE') | 
        (triplets_df['relation'] == 'DRUG_TARGET_PATHWAY')
    ]
    return drug_target_triplets


def get_one_hop_head_entity_neighbors(entity, triplets, max_degree=20):
    """Top max_degree one-hop neighbors for head entity based on degree."""
    entity_degrees = defaultdict(int)
    for head, relation, tail in triplets.values:
        entity_degrees[head] += 1
        entity_degrees[tail] += 1
    
    one_hop_neighbors = triplets[triplets['head'] == entity][['relation', 'tail']].values.tolist()
    sorted_neighbors = sorted(one_hop_neighbors, key=lambda x: entity_degrees[x[1]], reverse=True)
    top_neighbors = sorted_neighbors[:max_degree]
    
    return [f"{entity}-{rel}-{tail}" for rel, tail in top_neighbors]


def get_one_hop_tail_entity_neighbors(entity, triplets, max_degree=20):
    """Top max_degree one-hop neighbors for tail entity based on degree."""
    entity_degrees = defaultdict(int)
    for head, relation, tail in triplets.values:
        entity_degrees[head] += 1
        entity_degrees[tail] += 1
    
    one_hop_neighbors = triplets[triplets['tail'] == entity][['head', 'relation']].values.tolist()
    sorted_neighbors = sorted(one_hop_neighbors, key=lambda x: entity_degrees[x[0]], reverse=True)
    top_neighbors = sorted_neighbors[:max_degree]
    
    return [f"{head}-{rel}-{entity}" for head, rel in top_neighbors]


def save_evaluation_results(results, save_path):
    """Save final evaluation results as JSON."""
    import json
    os.makedirs(save_path, exist_ok=True)
    results_file = os.path.join(save_path, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {results_file}")