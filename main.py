import torch
import pandas as pd
import warnings

from utils import load_triplets
from model import get_tokenizer_and_model_classes
from train import train_and_evaluate

# ====================== CONFIG ======================
train_file_path = '/home/user/23h1710_KGC/PharmKG/train.txt'
valid_file_path = '/home/user/23h1710_KGC/PharmKG/valid.txt'
test_file_path  = '/home/user/23h1710_KGC/PharmKG/test.txt'
model_save_path = '/home/user/23h1710_KGC/PharmKG/with_sampl_r(PharmKG)_c1'

NUM_EPOCHS = 5
MODEL_NAME = "bert-base-uncased"
# ===================================================

# Ignore warnings (original behaviour)
warnings.filterwarnings("ignore")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

if __name__ == '__main__':
    # Load data
    train_triplets = load_triplets(train_file_path)
    valid_triplets = load_triplets(valid_file_path)
    test_triplets  = load_triplets(test_file_path)

    # Relation mapping
    relations = pd.concat([train_triplets['relation'],
                           valid_triplets['relation'],
                           test_triplets['relation']]).unique().tolist()
    relation_to_idx = {rel: idx for idx, rel in enumerate(relations)}

    # All triplets for neighbor lookup
    all_triplets = pd.concat([train_triplets, valid_triplets, test_triplets])

    # Get correct classes for the chosen model
    tokenizer_class, model_class = get_tokenizer_and_model_classes(MODEL_NAME)

    # Train & evaluate
    train_and_evaluate(
        model_name=MODEL_NAME,
        tokenizer_class=tokenizer_class,
        model_class=model_class,
        save_path=model_save_path,
        num_epochs=NUM_EPOCHS,
        train_triplets=train_triplets,
        valid_triplets=valid_triplets,
        test_triplets=test_triplets,
        relation_to_idx=relation_to_idx,
        all_triplets=all_triplets,
        device=device
    )