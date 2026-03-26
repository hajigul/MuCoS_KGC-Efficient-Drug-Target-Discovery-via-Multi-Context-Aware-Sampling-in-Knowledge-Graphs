from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)


def get_tokenizer_and_model_classes(model_name: str):
    """Returns tokenizer and model class (fixed to DistilBERT for this specific task)."""
    if "distilbert" in model_name.lower():
        return DistilBertTokenizer, DistilBertForSequenceClassification
    else:
        raise ValueError(f"Only DistilBERT is supported in this specific relation prediction version. Got: {model_name}")