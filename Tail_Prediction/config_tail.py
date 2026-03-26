import torch

# Device configuration
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using Device: {device}")

# Data paths (adjust to your local environment)
train_file_path = '/home/user/23h1710_KGC/KEGG50k/train.txt'
valid_file_path = '/home/user/23h1710_KGC/KEGG50k/valid_o.txt'
test_file_path = '/home/user/23h1710_KGC/KEGG50k/test_o.txt'
model_save_path = '/home/user/23h1710_KGC/KEGG50k/BERT_with_sampl_t_Prediction_dis'

# Model hyperparameters
MODEL_NAME = "distilbert-base-uncased"   # or "bert-base-uncased", "roberta-base"
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_LENGTH = 128
MAX_DEGREE_HEAD = 15      # n for head context
MAX_DEGREE_RELATION = 5   # k for relation context