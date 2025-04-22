from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

class NewsClassificationModel:
    """
    Initializes the RoBERTa model and tokenizer.
    """
    
    def __init__(self, model_name='roberta-base', num_labels=2):
        
        self.model_name = model_name
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def tokenize_data(self, data):
        """
        Tokenizes the input data.
        """

        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_data = data.map(tokenize_function, batched=True)
        return tokenized_data

    def get_model(self):
        """
        Returns the RoBERTa model.
        """

        return self.model

    def get_tokenizer(self):
        """
        Returns the tokenizer.
        """

        return self.tokenizer
