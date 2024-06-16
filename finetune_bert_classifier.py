import torch
import torch.nn.functional as F
import argparse
from transformers import BertForSequenceClassification
from load_data import load_bert_data
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import get_optimizer_and_scheduler, save_predictions, load_label_mapping

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='BERT Classifier training loop')

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    
    args = parser.parse_args()
    return args
    

def evaluate_model(model, dev_loader):
    model.eval()
    total_loss = 0
    valid_batches = 0

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Evaluating"):
            valid_labels = batch['labels'] < 10  # Boolean mask for labels within bounds
            if valid_labels.any():  # Only proceed if there are any valid labels
                inputs = {
                    'input_ids': batch['input_ids'][valid_labels].to(DEVICE),
                    'attention_mask': batch['attention_mask'][valid_labels].to(DEVICE),
                    'labels': batch['labels'][valid_labels].to(DEVICE)
                }
                outputs = model(**inputs)
                loss = outputs.loss
                total_loss += loss.item()
                valid_batches += 1

        return total_loss / valid_batches if valid_batches > 0 else 0



def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_score = float('-inf')
    model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
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
            labels = batch['labels'].to(DEVICE)

            outputs = model(**inputs)
            logits = outputs.logits
            
            loss = criterion(logits, labels)

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
        predictions = [np.argmax(score) if max(score) >= threshold else 10  # 'other' label
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
    plt.savefig('threshold_accuracy_plot.png')
    plt.close()


def main():
    args = get_args()
    train_loader, dev_loader, test_loader = load_bert_data(args.batch_size, args.test_batch_size)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)  # Excluding 'other'
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, train_loader)
    
    print("Training model")
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    model.load_state_dict(torch.load('best_model.pth'))
    
    print("Testing multiple thresholds on validation set")
    scores, _, labels = predict_with_confidence(model, dev_loader)
    threshold_range = np.linspace(0.3, 0.7, num=9)
    accuracies = test_thresholds(scores, labels, threshold_range)
    plot_thresholds(threshold_range, accuracies)
    best_threshold = threshold_range[np.argmax(accuracies)]
    print(f"Best threshold: {best_threshold}")

    print("Evaluating trained model on test set")
    scores, example_ids, labels = predict_with_confidence(model, test_loader)
    test_predictions = [np.argmax(score) if max(score) >= best_threshold else 10 for score in scores]
    test_accuracy = np.mean(np.array(test_predictions) == np.array(labels))
    print(f"Test Accuracy with best threshold {best_threshold}: {test_accuracy:.4f}")
    label_dict = load_label_mapping("label_dict.json")
    save_predictions(test_predictions, example_ids, labels, label_dict)

if __name__ == "__main__":
    main()
