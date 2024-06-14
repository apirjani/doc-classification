import torch
import argparse
from transformers import BertForSequenceClassification, BertTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from load_data import load_bert_data
import csv
import json
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='BERT Classifier training loop')

    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    
    args = parser.parse_args()
    return args
    
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

def evaluate_model(model, dev_loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Evaluating"):
            inputs = {
                'input_ids': batch['input_ids'].to(DEVICE),
                'attention_mask': batch['attention_mask'].to(DEVICE),
                'labels': batch['labels'].to(DEVICE)
            }
            outputs = model(**inputs)
            loss = outputs.loss

            total_loss += loss.item()

        return total_loss / len(dev_loader)


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    model.to(DEVICE)
    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()

            inputs = {
                'input_ids': batch['input_ids'].to(DEVICE),
                'attention_mask': batch['attention_mask'].to(DEVICE),
                'labels': batch['labels'].to(DEVICE)
            }
            outputs = model(**inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        eval_loss = evaluate_model(model, dev_loader)

        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {eval_loss:.4f}")

def predict(model, test_loader):
    model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(DEVICE),
                'attention_mask': batch['attention_mask'].to(DEVICE)
            }
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            ids.extend(batch['ids'])

    return predictions, ids

def compute_metrics(predictions, labels):
    # Compute accuracy over test set
    accuracy = (predictions == labels).mean()
    return accuracy

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
    return label_dict

def main():
    args = get_args()
    train_loader, dev_loader, test_loader = load_bert_data(args.batch_size, args.test_batch_size)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=11)
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, train_loader)
    print("Training model")
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    print("Evaluating model on test set")
    predictions, example_ids = predict(model, test_loader)
    labels = [batch['labels'] for batch in test_loader]
    accuracy = compute_metrics(predictions, labels)
    print(f"Test Accuracy: {accuracy:.4f}")
    label_dict = load_label_mapping("label_dict.json")
    save_predictions(predictions, example_ids, labels, label_dict)

if __name__ == "__main__":
    main()