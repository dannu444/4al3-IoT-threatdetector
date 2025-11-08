import torch
import torch.nn as nn
import torch.optim as optim

from argparse import ArgumentParser # use to implement command line args for running baseline vs model.

class GeneralNN(nn.Module):
    def __init__(self, input_dim: int, hidden_layer_dims: list[int], output_dim: int):
        super(GeneralNN, self).__init__()

        layers = []
        
        for i in range(0, len(hidden_layer_dims) + 1):
            if (i == 0):
                # first layer
                layers.append(nn.Linear(input_dim, hidden_layer_dims[i]))
                layers.append(nn.ReLU())
            elif (i == len(hidden_layer_dims)):
                # last layer
                layers.append(nn.Linear(hidden_layer_dims[i-1], output_dim))
            else:
                # middle layers
                layers.append(nn.Linear(hidden_layer_dims[i-1], hidden_layer_dims[i]))
                layers.append(nn.ReLU())
        
        self.fc_Network = nn.Sequential(*layers) # create fully connected network from all layers created.

    def forward(self, x):
        return self.fc_Network(x)

class MultiLogisticRegression(nn.Module):
    # for use in baseline majority vote
    # not yet implemented
    def __init__(self, input_dim, output_dim):
        super(MultiLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.Linear(x)
    
def calculate_full_loss(model, criterion, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y)
    model.train()

    return loss.item()

def calculate_accuracy(model, X, y):
    # not implemented!
    pass
    
def train_SGD(model, criterion, optimizer, X_train, y_train, X_val, y_val, iteration_num, batch_size, check_every):

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    iterations = []

    i = 0
    instances = X_train.shape[0]

    while (i <= iteration_num): 

        #TODO: Shuffle dataset every epoch.

        # train the model with batch
        optimizer.zero_grad()
        model_out = model(X_train[(i * batch_size) % instances : ((i * batch_size) % instances) + batch_size , :])
        loss = criterion(model_out, y_train[(i * batch_size) % instances : ((i * batch_size) % instances) + batch_size])
        loss.backward()
        optimizer.step()

        if (i % check_every == 0):
            # compute full loss and add to list to plot.
            train_losses.append(calculate_full_loss(model, criterion, X_train, y_train))
            val_losses.append(calculate_full_loss(model, criterion, X_val, y_val))
            train_accs.append(calculate_accuracy(model, X_train, y_train))
            val_accs.append(calculate_accuracy(model, X_val, y_val))
            iterations.append(i)

        i += 1

    return train_losses, val_losses, train_accs, val_accs, iterations
