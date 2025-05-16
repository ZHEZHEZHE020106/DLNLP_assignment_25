import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
import numpy as np
import torch
from sklearn.metrics import classification_report
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

def load_data(data_path):
    df = pd.read_csv(data_path, sep='\t')
    df.columns = ["text", "label", "id"]
    df["label"] = df["label"].apply(lambda x: list(map(int, str(x).split(","))))
    df["label_single"] = df["label"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else -1)
    return df

def map_data(df, label_map):
    reverse_map = {}
    for category, indices in label_map.items():
        for i in indices:
            reverse_map[i] = category

    # map 27 classes to 7 classes
    df["primary_label"] = df["label"].apply(lambda labels: reverse_map.get(labels[0], "other"))

    # keep only primary label
    df_filtered = df[df["primary_label"] != "other"].reset_index(drop=True)

    # map label to integer
    label_list = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
    label_to_id = {label: i for i, label in enumerate(label_list)}
    df_filtered["label"] = df_filtered["primary_label"].map(label_to_id)

    return df_filtered

def tokenize(tokenizer, batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=128)

def train_test_Split(df_filtered, tokenizer):
    # transfer to dataset from dataframe
    dataset = Dataset.from_pandas(df_filtered[["text", "label"]])

    # split train, val, test datasets
    df_final = df_filtered[["text", "label"]]

    # split test dataset at first
    df_train_val, df_test = train_test_split(
        df_final, test_size=0.2, stratify=df_final["label"], random_state=42
    )

    # split val dataset from train dataset
    df_train, df_val = train_test_split(
        df_train_val, test_size=0.1/0.8, stratify=df_train_val["label"], random_state=42
    )

    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)
    test_dataset = Dataset.from_pandas(df_test)

    train_dataset = train_dataset.map(lambda x: tokenize(tokenizer, x), batched=True)
    val_dataset   = val_dataset.map(lambda x: tokenize(tokenizer, x), batched=True)
    test_dataset  = test_dataset.map(lambda x: tokenize(tokenizer, x), batched=True)

    return train_dataset, val_dataset, test_dataset

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
        'f1_micro': f1_score(labels, preds, average='micro')
    }

def train(model, train_dataset, val_dataset, test_dataset, tokenizer):
    # hyperparameters for training

    training_args = TrainingArguments(
        output_dir="./roberta_goemotions_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
    )
    early_stop = EarlyStoppingCallback(early_stopping_patience=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
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

    # print precision, recall, F1 for each class
    report = classification_report(y_true, y_pred, digits=4, target_names=[
        "anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"
    ])
    print(report)

def main():
    data_path = "./datasets/train.tsv"
    data = load_data(data_path)

    label_map = {
        "anger":      [0, 1, 2],
        "disgust":    [3, 4],
        "fear":       [5, 6, 7],
        "joy":        [8, 9, 10, 11, 12, 13, 14],
        "sadness":    [15, 16, 17, 18],
        "surprise":   [19, 20],
        "neutral":    [27]
        }
    
    # 加载 tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    df = map_data(data, label_map)
    train_dataset, val_dataset, test_dataset = train_test_Split(df,tokenizer)

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=7)
    print("Start Training")
    train(model, train_dataset, val_dataset, test_dataset, tokenizer)
    print("Training Finished")


    model_path = "./pre_trained/RoBERTa_checkpoint"
    pre_trained_tokenizer = RobertaTokenizer.from_pretrained(model_path)
    pre_trained_model = RobertaForSequenceClassification.from_pretrained(model_path)
    evaluate(pre_trained_model, pre_trained_tokenizer, test_dataset)




if __name__ == "__main__":
    main()



