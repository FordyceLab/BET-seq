import pandas as pd
import numpy as np
import torch.utils.data
import matplotlib
import matplotlib.pyplot as plt
import copy


def clean_data(dataset_path):
    """
    Read in and zero-center data and produce a plot of 0-centered distribution
    
    Args:
    - dataset_path (str) - path to input dataset
    
    Returns:
    - cleaned dataset in pandas DataFrame format
    """
    
    # Read the Cbf1 dataset
    dataset = pd.read_table(dataset_path,
                            names=['sequence', 'input_count',
                                   'bound_count', 'ddG'])
    
    # Extract just the ddGs
    ddG = dataset['ddG']

    # Print the min, max, and mean values for the dataset
    print('Min: {:1.2f}, Max: {:1.2f}, Mean: {:1.2f}'.format(np.min(ddG),
                                                             np.max(ddG),
                                                             np.mean(ddG)))

    # Plot distribution
    plt.hist(ddG, 30, alpha=0.75)
    plt.xlabel(r'$\Delta \Delta G$')
    plt.ylabel('Count')
    plt.title(r'$\Delta \Delta G$ Distribution')
    plt.show()
    
    return dataset

def one_hot(seq):
    """
    One-hot encode a sequence

    Args:
    - seq (str): a string of the sequence to be encoded

    Returns:
    - (ndarray) A 4xN element numpy array containing the one-hot encoded sequence
    """

    # Make a matrix of zeros
    encoded_seq = np.zeros((4, len(seq)))

    # For each position in the length of the sequence
    for pos in range(len(seq)):

        # Set the appropriate row, col combo to 1
        nt = seq[pos]
        if nt == "A":
            encoded_seq[0, pos] = 1
        if nt == "C":
            encoded_seq[1, pos] = 1
        if nt == "G":
            encoded_seq[2, pos] = 1
        if nt == "T":
            encoded_seq[3, pos] = 1

    encoded_seq = encoded_seq.reshape((1, 40))

    # Return the encoded sequence
    return encoded_seq

def encode_seqs(sequences):
    """
    Encode all sequences in a one-hot fashion
    
    Args:
    - sequences (list-like) - all sequences to be encoded
    
    Returns:
    - SxL matrix of all one-hot encoded sequences (flattened)
    """
    
    seqs = np.concatenate([one_hot(seq) for seq in sequences])
    return seqs

class ddGDataset(torch.utils.data.Dataset):
    """
    Extend the PyTorch Dataset class to work with complete dataset prediction dataset
    """
    
    def __init__(self, data):
        """
        Instantiate a Dataset class
        
        Args:
        - data (ndarray) - one-hot encoded sequences in NxL format
        """
        super().__init__()
        self.data = torch.from_numpy(data)
    
    def __getitem__(self, index):
        seq = self.data[index, :-3]
        input_count = self.data[index, -3]
        bound_count = self.data[index, -2]
        ddG = self.data[index, -1]
        return (seq, input_count, bound_count, ddG)

    def __len__(self):
        return self.data.size()[0]

def process_data(dataset, batch_size, train_set_file, val_set_file, test_set_file):
    """
    Process data into training, validation, and test sets
    
    Args:
    - dataset (DataFrame): 2 column pandas dataframe with the first column named 'sequence'
        and the second named 'ratio'
    - batch_size (int): number of examples per batch
    - train_set_file (str) - path to file containing sequences to use for
        training
    - val_set_file (str) - path to file containing sequences to use for
        validation 
    - test_set_file (str) - path to file containing sequences to use for
        testing

    Returns:
    - tuple of the training, validation, and test set DataLoaders
    """

    train_seqs = pd.read_table(train_set_file, header=None)[0].tolist()
    val_seqs = pd.read_table(val_set_file, header=None)[0].tolist()
    test_seqs = pd.read_table(test_set_file, header=None)[0].tolist()
    
    # Get the complete processed data
    full_seq = np.array(dataset['sequence']).reshape((-1, 1))
    seqs = encode_seqs(dataset['sequence'])
    input_count = np.array(dataset['input_count']).reshape((-1, 1))
    bound_count = np.array(dataset['bound_count']).reshape((-1, 1))
    ddgs = np.array(dataset['ddG']).reshape((-1, 1))
    
    # Concatenate the flattened one-hot output and the measured ratio
    dataset = pd.DataFrame(np.hstack([full_seq, seqs, input_count, bound_count, ddgs]))
    
    # Get only the flanking sequences that should be in each of the three splits (allows for replicates)
    train = dataset[dataset[0].isin(set(train_seqs))].drop(dataset.columns[0], axis=1).values.astype('float')
    val = dataset[dataset[0].isin(set(val_seqs))].drop(dataset.columns[0], axis=1).values.astype('float')
    test = dataset[dataset[0].isin(set(test_seqs))].drop(dataset.columns[0], axis=1).values.astype('float')
    
    # Put the data into TensorDataset objects
    train_data = ddGDataset(train)
    val_data = ddGDataset(val)
    test_data = ddGDataset(test)
    
    # Assemble and return the DataLoader objects
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=True)
    
    return (train_loader, val_loader, test_loader)

class SingleTensorDataset(torch.utils.data.Dataset):
    """
    Extend the PyTorch Dataset class to work with complete dataset prediction
    dataset
    """
    
    def __init__(self, data):
        """
        Instantiate a Dataset class
        
        Args:
        - data (ndarray) - one-hot encoded sequences in NxL format
        """
        super().__init__()
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]
