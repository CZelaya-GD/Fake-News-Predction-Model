import pandas as pd
from src.data_loader import DataLoader
from src.model import NewsClassificationModel
from src.trainer import ModelTrainer

def main():
    """
    Main function to execute the fake news detection pipeline.
    """

    # Load and split data
    data_loader = DataLoader(data_path='evaluation.csv')  # Path to your data
    data = data_loader.load_data()
    if data is None:
        return

    train_data, test_data = data_loader.split_data(test_size=0.2)

    # Initialize model
    model_wrapper = NewsClassificationModel()
    model = model_wrapper.get_model()
    tokenizer = model_wrapper.get_tokenizer()

    # Tokenize data
    def tokenize_function(examples):

        return tokenizer(examples["text"], padding="max_length", truncation=True)

    train_dataset = train_data.map(tokenize_function, batched=True)
    test_dataset = test_data.map(tokenize_function, batched=True)

    # Train model
    trainer = ModelTrainer(model=model, train_dataset=train_dataset, eval_dataset=test_dataset)
    trainer.train(num_train_epochs=3)

    # Evaluate model
    trainer.evaluate()

if __name__ == "__main__":
    main()
