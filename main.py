# Import PyTorch and Hugging Face's transformers library
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


# Define a custom dataset class inheriting from PyTorch's Dataset class
class CustomDataset(Dataset):
    # Constructor to initialize the dataset with encodings and labels
    def __init__(self, encodings, labels):
        self.encodings = encodings  # Encoded input sequences
        self.labels = labels  # Corresponding labels for input sequences

    # Method to retrieve a single item at the specified index `idx`
    def __getitem__(self, idx):
        # Extract the encoding for the current index and convert to tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add the label for the current index to the item
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    # Method to get the number of items in the dataset
    def __len__(self):
        return len(self.labels)


# File paths for the training and validation CSV files
train_csv_file_path = 'CSV/TraindataTest1.csv'
val_csv_file_path = 'CSV/ValidationdataTest1.csv'


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
    overall_delta = {dimension: [] for dimension in quality_dimensions}

    evaluation_results = trainer.evaluate(test_dataset)
    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.tolist() 
    true_labels = test_dataset.labels

    # Print delta for each quality dimension
    for index, dimension in enumerate(quality_dimensions):
        pred_values = [pred[index] for pred in predicted_labels] 
        true_values = [true[index] for true in true_labels]

        delta = [pred - true for pred, true in zip(pred_values, true_values)]
        results[dimension]['delta'].extend(delta)
        results[dimension]['true_value'].extend(true_values)
        average_delta = sum(delta) / len(delta)
        print(f"{dimension}: {average_delta}")
        overall_delta[dimension].append(average_delta)

    excel_file_path = './results/results.xlsx'
    try:
        existing_df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        existing_df = None

    # Save the results to an Excel file
    df_results = pd.DataFrame()

    for dimension in quality_dimensions:
        df_results[f'{dimension}_delta'] = results[dimension]['delta']
        df_results[f'{dimension}_true_value'] = results[dimension]['true_value']
    
    overall_avg_delta = df_results[[f'{dimension}_delta' for dimension in quality_dimensions]].mean(axis=1)
    overall_avg_true_value = df_results[[f'{dimension}_true_value' for dimension in quality_dimensions]].mean(axis=1)
    
    df_results['overall_avg_delta'] = overall_avg_delta
    df_results['overall_avg_true_value'] = overall_avg_true_value
    if existing_df is not None:
        df_combined = pd.concat([existing_df, df_results], ignore_index=True)
    else:
        df_combined = df_results

    df_combined.to_excel(excel_file_path, index=False)

# Load a pre-trained BERT tokenizer and model from Hugging Face's model repository
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-german-uncased')
model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-german-uncased', num_labels=4)

# Load training and validation data from CSV files
train_texts, train_labels = load_dataset(train_csv_file_path)

val_texts, val_labels = load_dataset(val_csv_file_path)

# Tokenize the training and validation data
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Create dataset objects for training and validation sets
train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)

# Define training arguments for the Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",  # Perform evaluation at each logging step
)

# Initialize the Trainer with the model, training arguments, and datasets
trainer = Trainer(
    model=model,  # The pre-trained model to be fine-tuned
    args=training_args,  # Contains arguments for training (like batch size, learning rate, etc.)
    train_dataset=train_dataset,  # Dataset to use for training
    eval_dataset=val_dataset  # Dataset to use for evaluation
)

# Start training the model
trainer.train()

# Save the fine-tuned model to the specified directory
model.save_pretrained('my_finetuned_model/BertMultitaskModelStandard')

# File path for the test CSV file
test_csv_file_path = 'CSV/TestdataTest1.csv'

# Load test data from CSV file
test_texts, test_labels = load_dataset(test_csv_file_path)

# Tokenize the test data
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Create dataset object for test set
test_dataset = CustomDataset(test_encodings, test_labels)

# Evaluate the model on the test set
evaluation_results = trainer.evaluate(test_dataset)

# Usage:
runtimes = 1
for i in range(runtimes):
    evaluate_with_delta(trainer, test_dataset)
