import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import json

def format_data(directory):
    data = []
    labels = []
    ids = []  # To store unique identifiers for each text
    label_dict = {}
    current_id = 0  # Starting ID
    
    other_data = []
    other_labels = []
    other_ids = []
    
    # Get each category directory
    for index, category in enumerate(sorted(os.listdir(directory))):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            label_dict[category] = index
            # Load each file in the category directory
            for file_name in os.listdir(category_path):
                file_path = os.path.join(category_path, file_name)
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        if category == "other":  # Collect data for 'other' category separately
                            other_data.append(content)
                            other_labels.append(index)
                            other_ids.append(current_id)
                        else:
                            data.append(content)
                            labels.append(index)
                            ids.append(current_id)
                        current_id += 1  # Increment the ID for the next document
    
    # Shuffle the non-other data, labels, and ids in unison
    combined = list(zip(data, labels, ids))
    random.shuffle(combined)
    data, labels, ids = zip(*combined)
    
    return data, labels, ids, other_data, other_labels, other_ids, label_dict

def split_and_save(data, labels, ids, other_data, other_labels, other_ids, label_dict):
    # Split non-other data into train and temp sets (85% training, 15% split between validation and test)
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(data, labels, ids, test_size=0.15, random_state=42)
    
    # Combine the temp data with 'other' data for validation and test splits
    X_temp = list(X_temp) + other_data
    y_temp = list(y_temp) + other_labels
    ids_temp = list(ids_temp) + other_ids
    
    # Further split the combined data into validation and test sets
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(X_temp, y_temp, ids_temp, test_size=0.5, random_state=42)
    
    # Create DataFrames
    train_df = pd.DataFrame({'id': ids_train, 'text': X_train, 'label': y_train})
    val_df = pd.DataFrame({'id': ids_val, 'text': X_val, 'label': y_val})
    test_df = pd.DataFrame({'id': ids_test, 'text': X_test, 'label': y_test})
    
    # Save to CSV
    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('dev_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    
    # Optionally, save the label dictionary for reference
    with open('label_dict.json', 'w') as file:
        json.dump(label_dict, file)

# Main function to run the script
def main():
    directory = 'data'  # Path to the data directory
    data, labels, ids, other_data, other_labels, other_ids, label_dict = format_data(directory)
    split_and_save(data, labels, ids, other_data, other_labels, other_ids, label_dict)
    print("Data loading, shuffling, splitting, and saving completed.")

if __name__ == "__main__":
    main()
