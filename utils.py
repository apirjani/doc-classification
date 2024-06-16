import re
import torch
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import csv
import json

def preprocess_text(text):
    """
    Apply preprocessing steps to the input text.
    """
    # Remove non-word characters except for spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove email addresses
    text = re.sub(r'\b\S+@\S+\.\S+\b', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Collapse multiple spaces into a single space and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_optimizer_and_scheduler(args, model, train_loader):
    num_training_steps = args.num_epochs * len(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    if args.scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    elif args.scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    else:
        scheduler = None
    
    return optimizer, scheduler

def save_predictions(predictions, example_ids, labels, label_dict):
    # Save predictions to a CSV file
    # The CSV file should have two columns: "id" and "prediction" and "labels"
    with open('predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'prediction', 'label'])
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            writer.writerow([example_ids[i], label_dict[pred], label_dict[label]])

def load_label_mapping(file_path):
    with open(file_path, 'r') as file:
        label_dict = json.load(file)
    
    # we need the reverse mapping
    label_dict = {int(v): k for k, v in label_dict.items()}
    return label_dict