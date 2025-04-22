import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from transformers import Trainer, TrainingArguments

class ModelTrainer:
    """
    Handles model training and evaluation.
    """
    
    def __init__(self, model, train_dataset, eval_dataset, output_dir="./results"):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir

    def compute_metrics(self, eval_pred):
        """
        Computes the evaluation metrics (accuracy).
        """

        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": accuracy_score(labels, preds)}

    def train(self, num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=16, learning_rate=2e-5, weight_decay=0.01):
        """
        Trains the model using the Trainer API.
        """

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        self.trainer = trainer  # Store the trainer instance for later use
        return trainer

    def evaluate(self):
        """
        Evaluates the model and prints the classification report.
        """

        if not hasattr(self, 'trainer'):
            raise AttributeError("Model has not been trained yet. Call train() method first.")

        predictions = self.trainer.predict(self.eval_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        print(classification_report(self.eval_dataset['label'], preds))

    def get_trainer(self):
        """
        Returns the Trainer instance.
        """

        return self.trainer
