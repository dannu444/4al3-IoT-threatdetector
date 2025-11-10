import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

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
        return self.linear(x)
    
def calculate_full_loss(model, criterion, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y)
    model.train()

    return loss.item()

def calculate_f_score(model, X, y):
    # get predictions
    outputs = None
    model.eval()
    with torch.no_grad():
        outputs = model(X)
    model.train()

    y_true = torch.argmax(y, dim=1).numpy()
    y_pred = torch.argmax(outputs, dim=1).numpy()

    return f1_score(y_true, y_pred, average="macro")

def train_SGD(model, criterion, optimizer, X_train, y_train, X_val, y_val, iteration_num, batch_size, check_every):

    # init all returns
    train_losses = []
    val_losses = []
    train_fs = []
    val_fs = []
    iterations = []

    # init vars for use within train loop
    i = 0
    instances = X_train.shape[0]

    # create copies of train datasets for shuffling
    X_train_shuf = X_train.clone()
    y_train_shuf = y_train.clone()

    while (i <= iteration_num): 

        # shuffle dataset roughly every epoch
        if (i % (instances // batch_size) == 0):
            X_train_np, y_train_np = shuffle(X_train_shuf.numpy(), y_train_shuf.numpy())
            X_train_shuf = torch.tensor(X_train_np, dtype=torch.float32)
            y_train_shuf = torch.tensor(y_train_np, dtype=torch.float32)

        # train the model with batch
        optimizer.zero_grad()
        model_out = model(X_train_shuf[(i * batch_size) % instances : ((i * batch_size) % instances) + batch_size , :])
        loss = criterion(model_out, y_train_shuf[(i * batch_size) % instances : ((i * batch_size) % instances) + batch_size])
        loss.backward()
        optimizer.step()

        if (i % check_every == 0):
            # compute full loss and add to list to plot.
            train_losses.append(calculate_full_loss(model, criterion, X_train, y_train))
            val_losses.append(calculate_full_loss(model, criterion, X_val, y_val))
            train_fs.append(calculate_f_score(model, X_train, y_train))
            val_fs.append(calculate_f_score(model, X_val, y_val))
            iterations.append(i)

        i += 1

    return train_losses, val_losses, train_fs, val_fs, iterations

def main():
    parser = ArgumentParser()
    parser.add_argument("input_path", type=str, help="Relative path to input data (csv file).")
    parser.add_argument("target_path", type=str, help="Relative path to target data (csv file).")
    args = parser.parse_args()

    # Get input and target data
    input = pd.read_csv(args.input_path)
    target = pd.read_csv(args.target_path)

    # Drop unnecessary features
    input.drop("bwd_URG_flag_count", axis=1, inplace=True)

    # Transfer input and target data to numpy arrays
    input = input.to_numpy()
    target = target.to_numpy()

    # Shuffle input and target
    shuffle_index = np.random.permutation(input.shape[0])
    input = input[shuffle_index]
    target = target[shuffle_index]

    # Split data into training and validation
    X_train = input[:80000, :]
    X_val = input[80000:, :]
    y_train = target[:80000, :]
    y_val = target[80000:, :]

    # Create tensors for data
    X_train_t = torch.tensor(X_train).float()
    X_val_t = torch.tensor(X_val).float()
    y_train_t = torch.tensor(y_train).float()
    y_val_t = torch.tensor(y_val).float()

    # Print shape of tensors
    print(f"Training features shape: {X_train_t.shape}")
    print(f"Validation features shape: {X_val_t.shape}")
    print(f"Training target shape: {y_train_t.shape}")
    print(f"Validation target shape: {y_val_t.shape}")

    model = GeneralNN(X_train.shape[1], [128, 64, 32, 16], y_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses, val_losses, train_fs, val_fs, iterations = train_SGD(model, criterion, optimizer, X_train_t, y_train_t, X_val_t, y_val_t, 100000, 16, 1000)

    # Initial plotting
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, train_losses, label='Training Loss', marker='o')
    plt.plot(iterations, val_losses, label='Validation Loss', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
