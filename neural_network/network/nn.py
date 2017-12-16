import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from tqdm import *

dtype = torch.cuda.FloatTensor

class LBRD(nn.Module):
    """
    Create a sandwich layer for linear-batchnorm-ReLU-dropout

    Args:
    - input_dim (int) - input size to linear layer
    - output_dim (int) - output size of linear layer
    - dropout (float) - dropout probability for layer
    """
    def __init__(self, input_dim, output_dim, dropout):
        super(LBRD, self).__init__()
        
        # Linear transform
        self.fc = nn.Linear(input_dim, output_dim)
        
        # Xavier initialization for linear transform
        init.xavier_uniform(self.fc.weight)
        
        # Batchnorm
        self.batchnorm = torch.nn.BatchNorm1d(output_dim)
        
        # Dropout
        self.dropout = torch.nn.Dropout(p=dropout)
        
    def forward(self, x):
        """
        Perform forward pass

        Args:
        - x (array) - input to layer

        Returns:
        - layer output
        """
        
        # Forward pass on x
        x = self.fc(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        out = self.dropout(x)
        return out

class Net(nn.Module):
    """
    Network of two hidden layers and linear transform on the output
    """
    
    def __init__(self):
        super(Net, self).__init__()
        
        # Two hidden layers
        self.layer1 = LBRD(40, 500, 0)
        self.layer2 = LBRD(500, 500, 0.25)
        self.layer3 = LBRD(500, 250, 0.25)
        
        # Linear transform
        self.linear = nn.Linear(250, 1)

    def forward(self, x):
        
        # Forward pass
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.linear(x)
        return out
    
def en_loss(output, target, lambda_val=0.5):
    """
    Elastic net loss using PyTorch loss functions
    
    Args:
    - output (Variable) - output from neural net model
    - target (Variable) - true values sequences
    - lambda (float) - mixing factor for L1/L2 (lambda * L2 + (1 - labmda) * L1)
    
    Returns:
    - elastic net loss
    """

    MSE_fn = torch.nn.MSELoss(size_average=False)
    L1_fn = torch.nn.L1Loss(size_average=False)
    return lambda_val * MSE_fn(output, target) + (1 - lambda_val) * L1_fn(output, target)

def weighted_MSE(output, target, in_count, bound_count, weighted=True, mean=True):
    """
    Poisson-error weighted mean squared error loss function
    
    Args:
    - target (array) - array of target values
    - in_count (array) - input library read counts
    - bound_count (array) - input library read counts
    - weighted (bool) - whether to weight the loss (default = True)
    - mean (bool) - return mean vs. summed loss (default = True)
    
    Returns:
    - loss
    """

    count_frac_error = ((in_count.sqrt() / in_count)**2 +
                        (bound_count.sqrt() / bound_count)**2).sqrt()
    
    loss_contrib = 1 - count_frac_error
    
    if weighted:
        error = loss_contrib * (output - target)**2
    else:
        error = (output - target)**2
    
    if mean:
        error = torch.mean(error)
    else:
        error = torch.sum(error)
    
    return error

def train(model, train_loader, val_loader, complete_loader, optimizer):
    """
    Train the model for a single epoch
    
    Args:
    - model (Net) - model to train
    - train_loader (DataLoader) - training data wrapped in DataLoader class
    - val_loader (DataLoader) - validation data wrapped in DataLoader class
    - complete_loader (DataLoader) - sequences to be predicted wrapped in DataLoader class
    arising from SingleTensorDataset
    - optimizer (Optim) - optimizer for training

    Returns:
    - tuple of the training and validation loss values
    """
    
    # Put the model in training mode
    model.train()
    
    # For each minibatch
    for batch_id, (data, in_count, bound_count, target) in enumerate(tqdm(train_loader,
                                                                          position=0,
                                                                          unit='batch',
                                                                          leave=True)):
        
        # Put the data on the gpu
        data = Variable(data).type(dtype)
        in_count = Variable(in_count).type(dtype)
        bound_count = Variable(bound_count).type(dtype)
        target = Variable(target).type(dtype)
        target += Variable(torch.zeros(target.size()).normal_(0, 0.5).cuda())
        
        # Zero the gradients in the optimizer
        optimizer.zero_grad()
        model.zero_grad()
        
        # Get predicted values
        output = model(data).type(dtype)
        
        # Get the loss
        loss = weighted_MSE(output, target, in_count, bound_count, weighted=False)
        
        # Perform backprop
        loss.backward()
        
        # Take a step
        optimizer.step()
    
    # 
    train_loss = test(model, train_loader)
    val_loss = test(model, val_loader)
    print('Train Loss = {}, Val Loss = {}'.format(round(train_loss, 5), round(val_loss, 5)))
    show_all_preds(model, complete_loader)
    return train_loss, val_loss

def test(model, loader):
    """
    Test model predictions on a dataset
    
    Args:
    - model (Net) - model to test
    - loader (DataLoader) - DataLoader with examples and targets
    
    Returns:
    - test set average loss
    """
    
    # Set model into evaluation mode
    model.eval()
    
    # Set test loss to 0
    test_loss = 0
    
    # Set the number of observed targets to 0
    num_targets = 0
    
    # For each dataset in the loader
    for data, in_count, bound_count, target in loader:
        
        # Convert to Variables and put on GPU
        data = Variable(data, volatile=True).type(dtype)
        in_count = Variable(in_count, volatile=True).type(dtype)
        bound_count = Variable(bound_count, volatile=True).type(dtype)
        target = Variable(target, volatile=True).type(dtype)
        
        # Get predicted values
        output = model(data).type(dtype)
        
        # Add number of of observed sequences
        num_targets += target.size()[0]
        
        # Add the test loss
        test_loss += weighted_MSE(output, target, in_count, bound_count, weighted=False, mean=False)
    
    # Get and return average test loss
    test_loss = float(np.sqrt(test_loss.cpu().data.numpy() / num_targets))
    return test_loss

def show_all_preds(model, loader):
    """
    Plot predicted values for all sequences

    Args:
    - model (Net) - model to test
    - loader (DataLoader) - DataLoader with examples and targets
    """
    
    # Set model into evaluation mode
    model.eval()
    
    # Make an array to hold the output
    complete_output = np.array([])
    
    # For each sequence batch
    for data in loader:
        
        # Put data onto GPU
        data = Variable(data, volatile=True).type(dtype)
        
        # Get the output and drop ones that are 0 or less
        output = model(data)
        output = output.cpu().data.numpy()
        
        # Convert to ddGs and add to output array
        complete_output = np.append(complete_output, output)

    # Print some stats
    print('Min: {:1.2f}, Max: {:1.2f}, Mean: {:1.2f}'.format(np.min(complete_output),
                                                             np.max(complete_output),
                                                             np.mean(complete_output)))
        
    # Make a histogram of the predicted values
    plt.hist(complete_output, 30, alpha=0.75)
    plt.xlabel(r'$\Delta \Delta G$')
    plt.ylabel('Count')
    plt.show()
    
def train_nn(train_loader, val_loader, prediction_loader, acc_file,
             max_epochs=100, early_stop=5, lr=1e-3, L2=0.01, min_improvement=1e-4):
    """
    Train a model
    
    Args:
    - train_loader (DataLoader) - training data wrapped in DataLoader class
    - val_loader (DataLoader) - validation data wrapped in DataLoader class
    - prediction_loader (DataLoader) - sequences to be predicted wrapped in
        DataLoader class
    - acc_file (str) - path to file to use for logging training and validation
        losses
    - max_epochs (int) - maximum number of training epochs (default = 100)
    - early_stop (int) - number epochs to allow training without improvement on
        validation dataset (default = 5)
    - lr (float) - learning rate (default = 1e-3)
    - L2 (float) - L2 weight decay (default = 0.01)
    - min_improvement (float) - minimum improvement for new parameters to be
        considered "better" than last epoch (default = 1e-4)

    Returns:
    - model with best parameters
    """
    
    # Instantiate the network and shift it to the GPU
    model = Net()
    model.cuda()

    # Instantiate and SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=L2)
    next_lr = lr / 10

    # Start the counter for early stopping
    counter = early_stop
    
    # Set the minimum observed validation loss to infinity
    min_val_loss = np.inf
    
    with open(acc_file, 'w') as acc_handle:
        acc_handle.write('epoch\ttrain_acc\tval_acc\n')
        
        # Train
        for epoch in range(max_epochs):
            # Get the val loss
            train_loss, val_loss = train(model, train_loader, val_loader, prediction_loader, optimizer)
            
            acc_handle.write(str(epoch) + '\t' + str(train_loss) + '\t' + str(val_loss) + '\n')
            
            # Determine if this is the best model seen
            if min_val_loss - val_loss  > min_improvement:
                counter = early_stop
                min_val_loss = val_loss
                best_model = model.state_dict()
                print("Best model!")
            else:
                counter -= 1

            # Break for early stopping when counter reaches 0
            if counter == np.floor(early_stop / 2):
                optimizer = optim.SGD(model.parameters(), lr=next_lr, weight_decay=L2)
                next_lr /= 10
                print("Learning rate decayed 10x")

            if counter == 0:
                break
    
    # Return the best model from training
    model = Net()
    model.load_state_dict(best_model)
    model = model.cuda()
    return model

def make_predictions(parameter_file, prediction_loader):
    """
    Load a model using stored parameters and output predictions

    Args:
    - parameter_file (str) - Path to parameter file
    - prediction_loader (str) - DataLoader containing input to be predicted
    """

    # Instantiate the model, load the parameters, and place on GPU
    model = Net()
    model.load_state_dict(torch.load(parameter_file))
    model = model.cuda()

    # Switch model to evaluation mode
    model.eval()

    # Create an array for the output
    complete_output = np.array([])

    # Predict all the input data
    for data in prediction_loader:
        data = Variable(data, volatile=True).type(dtype)
        output = model(data).cpu().data.numpy()
        complete_output = np.append(complete_output, output)
    
    # Center the predictions since mean of ddG distribution is 0 by definition
    centered_complete_output = complete_output - np.mean(complete_output)

    # Return the predicted values
    return(centered_complete_output)

