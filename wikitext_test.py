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

    # Tokenize the dataset (optional)
    tokenizer = get_tokenizer('basic_english')


    # Tokenize the dataset (optional)
    tokenizer = get_tokenizer('basic_english')
    train_tokenized = [tokenizer(line) for line in train_dataset]
    valid_tokenized = [tokenizer(line) for line in valid_dataset]
    test_tokenized = [tokenizer(line) for line in test_dataset]


    # Print some sample data
    print("Sample train data:", train_tokenized[:5])
    print("Sample valid data:", valid_tokenized[:5])
    print("Sample test data:", test_tokenized[:5]) 