import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import json

def format_data(directory):
    data = []
    labels = []
    ids = []  
    other_data = []
    other_labels = []
    other_ids = []
    label_dict = {}
    current_id = 0
    
    # Collect all categories except 'other'
    categories = [d for d in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, d)) and d != "other"]
    # Assign indices to these categories
    label_dict = {category: i for i, category in enumerate(categories)}
    # Handle 'other' separately
    other_label_index = len(categories)  # This will be '10' if there are 10 other categories

    # Process all files in the directory
    for category in categories + ['other']:
        category_path = os.path.join(directory, category)
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if category == "other":
                        other_data.append(content)
                        other_labels.append(other_label_index)  # Use '10' for 'other'
                        other_ids.append(current_id)
                    else:
                        data.append(content)
                        labels.append(label_dict[category])
                        ids.append(current_id)
                    current_id += 1

    random.shuffle(list(zip(data, labels, ids)))  # Shuffle the non-other data
    
    return data, labels, ids, other_data, other_labels, other_ids, label_dict, other_label_index

def split_and_save(data, labels, ids, other_data, other_labels, other_ids, label_dict, other_label_index):
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(data, labels, ids, test_size=0.15, random_state=42)
    X_temp.extend(other_data)
    y_temp.extend(other_labels)
    ids_temp.extend(other_ids)

    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(X_temp, y_temp, ids_temp, test_size=0.5, random_state=42)

    train_df = pd.DataFrame({'id': ids_train, 'text': X_train, 'label': y_train})
    val_df = pd.DataFrame({'id': ids_val, 'text': X_val, 'label': y_val})
    test_df = pd.DataFrame({'id': ids_test, 'text': X_test, 'label': y_test})

    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('dev_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)

    # Save the label dictionary including 'other'
    label_dict['other'] = other_label_index
    with open('label_dict.json', 'w') as file:
        json.dump(label_dict, file)

def main():
    directory = 'data'  # Specify the data directory
    data, labels, ids, other_data, other_labels, other_ids, label_dict, other_label_idx = format_data(directory)
    split_and_save(data, labels, ids, other_data, other_labels, other_ids, label_dict, other_label_idx)
    print("Data loading, shuffling, splitting, and saving completed.")

if __name__ == "__main__":
    main()
