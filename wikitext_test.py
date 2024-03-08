from datasets import load_dataset

if __name__ == '__main__':
    # Download the dataset
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    # Save the dataset to a data folder
    dataset.save_to_disk('./data/wikitext-103')

    # Accessing the train, test, and validation splits
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    validation_dataset = dataset['validation']

    # Print some sample data
    print("Sample train data:", train_dataset[:5])
    print("Sample valid data:", validation_dataset[:5])
    print("Sample test data:", test_dataset[:5]) 