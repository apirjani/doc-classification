import torch
import torch.nn.functional as F
import argparse
from transformers import BertForSequenceClassification, BertTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from load_data import load_bert_data
import csv
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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
    parser.add_argument('--learning_rate', type=float, default=3e-4)
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
    best_score = float('-inf')
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
        score = -eval_loss
        
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {eval_loss:.4f}")

def predict(model, test_loader):
    model.eval()
    predictions = []
    ids = []
    labels = []

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
            ids.extend(batch['ids'].cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())

    return predictions, ids, labels

def compute_metrics(predictions, labels):
    # Compute accuracy over test set

    accuracy = np.mean(np.array(predictions) == np.array(labels))
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
    
    # we need the reverse mapping
    label_dict = {int(v): k for k, v in label_dict.items()}
    return label_dict

def predict_with_confidence(model, loader):
    model.eval()
    all_scores = []
    all_labels = []
    all_ids = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = {
                'input_ids': batch['input_ids'].to(DEVICE),
                'attention_mask': batch['attention_mask'].to(DEVICE)
            }
            outputs = model(**inputs)
            logits = outputs.logits
            scores = F.softmax(logits, dim=1)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_ids.extend(batch['ids'].cpu().numpy())
    
    return all_scores, all_ids, all_labels

def test_thresholds(scores, labels, threshold_range):
    accuracies = []
    for threshold in threshold_range:
        predictions = [np.argmax(score) if max(score) >= threshold else 6  # 'other' label
                       for score in scores]
        accuracy = np.mean(np.array(predictions) == np.array(labels))
        accuracies.append(accuracy)
    return accuracies

def plot_thresholds(threshold_range, accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(threshold_range, accuracies, marker='o')
    plt.title('Accuracy vs Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def main():
    args = get_args()
    train_loader, dev_loader, test_loader = load_bert_data(args.batch_size, args.test_batch_size)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)  # Excluding 'other'
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, train_loader)
    
    print("Training model")
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    model.load_state_dict(torch.load('best_model.pth'))
    
    print("Testing multiple thresholds on validation set")
    scores, labels = predict_with_confidence(model, dev_loader)
    threshold_range = np.linspace(0.5, 0.9, num=9)
    accuracies = test_thresholds(scores, labels, threshold_range)
    plot_thresholds(threshold_range, accuracies)

    print("Evaluating trained model on test set")
    scores, example_ids, labels = predict_with_confidence(model, test_loader)
    best_threshold = threshold_range[np.argmax(accuracies)]
    test_predictions = [np.argmax(score) if max(score) >= best_threshold else 10 for score in scores]
    test_accuracy = np.mean(np.array(test_predictions) == np.array(labels))
    print(f"Test Accuracy with best threshold {best_threshold}: {test_accuracy:.4f}")
    label_dict = load_label_mapping("label_dict.json")
    save_predictions(test_predictions, example_ids, labels, label_dict)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()