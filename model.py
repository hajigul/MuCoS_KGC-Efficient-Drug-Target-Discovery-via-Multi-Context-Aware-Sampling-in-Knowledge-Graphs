from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification
)


def get_tokenizer_and_model_classes(model_name: str):
    """Returns the correct tokenizer and model class for the given model name."""
    name = model_name.lower()
    if "bert-base-uncased" in name or "bert" in name:
        return BertTokenizer, BertForSequenceClassification
    elif "roberta" in name:
        return RobertaTokenizer, RobertaForSequenceClassification
    elif "distilbert" in name:
        return DistilBertTokenizer, DistilBertForSequenceClassification
    else:
        raise ValueError(f"Unsupported model: {model_name}")