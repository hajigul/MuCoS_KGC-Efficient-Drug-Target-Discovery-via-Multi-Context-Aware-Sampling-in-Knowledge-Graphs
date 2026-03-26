import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from dataset_tail import load_triplets
from train_tail import train_and_evaluate
import config_tail

def main():
    # Load triplets
    train_triplets = load_triplets(config_tail.train_file_path)
    valid_triplets = load_triplets(config_tail.valid_file_path)
    test_triplets = load_triplets(config_tail.test_file_path)

    # Unique entities in the knowledge graph (from all datasets)
    # Note: The original code used only tails; we keep that for consistency.
    entities = pd.concat([train_triplets['tail'],
                          valid_triplets['tail'],
                          test_triplets['tail']]).unique().tolist()
    entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}

    # Combine all triplets for neighbor extraction
    all_triplets = pd.concat([train_triplets, valid_triplets, test_triplets])

    # Train and evaluate the model
    train_and_evaluate(
        model_name=config_tail.MODEL_NAME,
        tokenizer_class=DistilBertTokenizer,
        model_class=DistilBertForSequenceClassification,
        save_path=config_tail.model_save_path,
        num_epochs=config_tail.NUM_EPOCHS,
        train_triplets=train_triplets,
        valid_triplets=valid_triplets,
        test_triplets=test_triplets,
        all_triplets=all_triplets,
        entity_to_idx=entity_to_idx,
        batch_size=config_tail.BATCH_SIZE,
        learning_rate=config_tail.LEARNING_RATE,
        max_length=config_tail.MAX_LENGTH,
        max_degree_head=config_tail.MAX_DEGREE_HEAD,
        max_degree_relation=config_tail.MAX_DEGREE_RELATION,
        device=config_tail.device
    )

if __name__ == '__main__':
    main()