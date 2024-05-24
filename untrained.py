import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


# Define a custom dataset class inheriting from PyTorch's Dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Function to read a CSV file and extract texts and labels
def load_dataset(csv_file_path):
    df = pd.read_csv(csv_file_path, delimiter=';', encoding='utf-8')
    texts = df['sentence'].tolist()
    for col in ['precision', 'complexity', 'clearness', 'grammaticality']:
        df[col] = df[col].str.replace(',', '.').astype(float)
    labels = df[['precision', 'complexity', 'clearness', 'grammaticality']].values.tolist()
    return texts, labels

def evaluate_with_delta(trainer, test_dataset):
    quality_dimensions = ['precision', 'complexity', 'clearness', 'grammaticality']
    results = {dimension: {'delta': [], 'true_value': []} for dimension in quality_dimensions}

    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.tolist()
    true_labels = test_dataset.labels

    for index, dimension in enumerate(quality_dimensions):
        pred_values = [pred[index] for pred in predicted_labels]
        true_values = [true[index] for true in true_labels]

        delta = [pred - true for pred, true in zip(pred_values, true_values)]
        results[dimension]['delta'].extend(delta)
        results[dimension]['true_value'].extend(true_values)
        average_delta = sum(delta) / len(delta)
        print(f"{dimension}: {average_delta}")

    excel_file_path = './results/results.xlsx'
    df_results = pd.DataFrame()

    for dimension in quality_dimensions:
        df_results[f'{dimension}_delta'] = results[dimension]['delta']
        df_results[f'{dimension}_true_value'] = results[dimension]['true_value']

    overall_avg_delta = df_results[[f'{dimension}_delta' for dimension in quality_dimensions]].mean(axis=1)
    overall_avg_true_value = df_results[[f'{dimension}_true_value' for dimension in quality_dimensions]].mean(axis=1)

    df_results['overall_avg_delta'] = overall_avg_delta
    df_results['overall_avg_true_value'] = overall_avg_true_value
    df_results.to_excel(excel_file_path, index=False)

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# Load and tokenize test data
test_csv_file_path = 'CSV/TestdataTest1.csv'
test_texts, test_labels = load_dataset(test_csv_file_path)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
test_dataset = CustomDataset(test_encodings, test_labels)

# Define training arguments and initialize the Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=0,  # No training epochs
    per_device_eval_batch_size=64,
    do_train=False,  # Disable training
    do_eval=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None,  # No training dataset
    eval_dataset=test_dataset
)

# Evaluate the model on the test set without training
runtimes = 1
for i in range(runtimes):
    evaluate_with_delta(trainer, test_dataset)



