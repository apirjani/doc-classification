from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification
import pandas as pd
import torch


class BERTClassifierDataset(Dataset):

    def __init__(self, split):
        self.split = split
        self.data = self.load_data()
        self.labels = self.data['label'].values
        self.ids = self.data['id'].values

    def preprocess(self, data):
        # Remove non-word characters, convert to lowercase, and trim whitespace
        data['text'] = data['text'].str.replace('[^\w\s]', '', regex=True).str.lower().str.strip()
        # Remove email addresses and URLs from the text
        data['text'] = data['text'].str.replace(r'\b\S+@\S+\.\S+\b', '', regex=True).replace(r'http\S+', '', regex=True)
        # Remove numbers and collapse multiple spaces into a single space
        data['text'] = data['text'].str.replace(r'\d+', '', regex=True).replace(r'\s+', ' ', regex=True)

        return data

    
    def load_data(self):
        data = pd.read_csv(f'{self.split}_data.csv')
        return self.preprocess(data)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.data.items()}
        item['label'] = self.labels[idx]
        item['id'] = self.ids[idx]
        return item
    
def create_collate_fn(tokenizer):
    def collate_fn(batch):
        ids = torch.tensor([item['id'] for item in batch])
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch])
        # Tokenize and pad in one pass
        encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return {**encodings, 'labels': labels, 'ids': ids}
    return collate_fn

def get_dataloader(batch_size, split, tokenizer):
    dset = BERTClassifierDataset(split)
    collate_fn = create_collate_fn(tokenizer)
    dataloader = DataLoader(dset, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader

def load_bert_data(batch_size, test_batch_size):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_loader = get_dataloader(batch_size, "train", tokenizer)
    dev_loader = get_dataloader(test_batch_size, "dev", tokenizer)
    test_loader = get_dataloader(test_batch_size, "test", tokenizer)
    
    return train_loader, dev_loader, test_loader

if __name__ == "__main__":
    train_loader, dev_loader, test_loader = load_bert_data(16, 16)
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(dev_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    print("Data loading completed.")