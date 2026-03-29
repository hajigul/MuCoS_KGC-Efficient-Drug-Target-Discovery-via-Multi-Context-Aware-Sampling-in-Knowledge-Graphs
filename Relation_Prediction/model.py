from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification
)



# ==================== MODEL & TOKENIZER LOADER ====================
# This block imports pre-trained transformer models and their corresponding tokenizers
# from the HuggingFace Transformers library. These models (BERT, RoBERTa, DistilBERT)
# are widely used for NLP tasks. In this project, they are used for relation prediction,
# where the model takes a text input and classifies it into one of the relation labels.


# ==================== FUNCTION: GET MODEL AND TOKENIZER ====================
# This function dynamically loads a pre-trained model and its tokenizer based on
# the given model name. It allows flexibility to switch between different transformer
# architectures without changing the rest of the code.
#
# The function also sets the number of output labels (num_labels), which corresponds
# to the number of relations in the knowledge graph. This ensures the model is properly
# configured for multi-class classification.

def get_model_and_tokenizer(model_name, num_labels):
    """Load appropriate model and tokenizer based on model name."""


        # ==================== BERT MODEL ====================
    # If the selected model is BERT, load the BERT tokenizer and classification model.
    # BERT is a bidirectional transformer that captures context from both directions.
    if model_name == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


        # ==================== ROBERTA MODEL ====================
    # If RoBERTa is selected, load its tokenizer and model.
    # RoBERTa is an improved version of BERT with better training strategies
    # and often achieves higher performance on many NLP tasks.
    elif model_name == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


        # ==================== DISTILBERT MODEL ====================
    # If DistilBERT is selected, load its tokenizer and model.
    # DistilBERT is a smaller and faster version of BERT that retains most
    # of its performance while being more efficient.
    elif model_name == "distilbert-base-uncased":
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

            # ==================== ERROR HANDLING ====================
    # If the provided model name is not supported, raise an error.
    # This ensures the user selects only from predefined valid models.
    
    else:
        raise ValueError(f"Model {model_name} not supported. Choose from: bert-base-uncased, roberta-base, distilbert-base-uncased")
    
        # ==================== RETURN ====================
    # Return both the model and tokenizer so they can be used in training and inference.
    
    return model, tokenizer