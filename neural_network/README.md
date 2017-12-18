# Neural network analysis

This repository contains all the code to train and perform the neural network analyses described in the *Le, et al.* manuscript. This repository contains the following files:

- `train_models.ipynb` - Master Jupyter notebook for training all networks
- `models`
    - `cbf1_model.acc` - Log file containing accuracies for Cbf1 training and validation sets across all training epochs
    - `cbf1_model_params.torch` - Cbf1 network parameters
    - `pho4_model.acc` - Log file containing accuracies for Pho4 training and validation sets across all training epochs
    - `pho4_model_params.torch` - Pho4 network parameters
- `network`
    - `data_utils.py` - Python classes and functions to handle loading raw data
    - `nn.py` - Python classes and utilities to create and train the network models
