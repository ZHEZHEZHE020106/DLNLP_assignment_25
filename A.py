import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
import numpy as np
import torch
from sklearn.metrics import classification_report

def load_data(data_path):
    df = pd.read_csv(data_path)
    df["label"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

    return df

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_micro": f1_score(labels, preds, average="micro"),
    }

def train_test_Split(df, tokenizer):
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(), df["label"].tolist(), test_size=0.1, random_state=42
        )
    
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

    train_dataset = train_dataset.map(lambda x: {**tokenize(tokenizer, x), "label": x["label"]}, batched=True)
    val_dataset = val_dataset.map(lambda x: {**tokenize(tokenizer, x), "label": x["label"]}, batched=True)

    return train_dataset, val_dataset



def tokenize(tokenizer, batch):
    
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=128)

def train(model, train_dataset, val_dataset, tokenizer):
    training_args = TrainingArguments(
        output_dir="./bert_sentiment140_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10, 
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
        )

    early_stop = EarlyStoppingCallback(early_stopping_patience=2)

    #Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[early_stop]
        )

    trainer.train()

def evaluate(model, tokenizer, test_dataset):
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
        )
    predictions = trainer.predict(test_dataset)
    
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    # report precision, recall, F1 for each class
    report = classification_report(y_true, y_pred, digits=4, target_names=[
        "negative", "positive"
    ])
    print(report)


def main():
    # Train Model
    data_path = "./datasets/sentiment140_sampled_10000_each.csv"
    data = load_data(data_path)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset, val_dataset = train_test_Split(data, tokenizer)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    print("Start Training")
    train(model, train_dataset, val_dataset, tokenizer)
    print("Training Finished")

    # Load Model for Evaluation
    model_path = "./pre_trained/Bert_checkpoint"
    pre_trained_tokenizer = BertTokenizer.from_pretrained(model_path)
    pre_trained_model = BertForSequenceClassification.from_pretrained(model_path)
    test_path = "./datasets/sentiment140_test_2000.csv"

    test_data = load_data(test_path)
    test_dataset = Dataset.from_pandas(test_data[["text", "label"]])
    test_dataset = test_dataset.map(lambda x: {**tokenize(pre_trained_tokenizer, x), "label": x["label"]}, batched=True)

    print("Start Evaluation")
    evaluate(pre_trained_model, pre_trained_tokenizer, test_dataset)
    print("Evaluation Finished")

if __name__ == "__main__":
    main()