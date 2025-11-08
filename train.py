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
    

    

    
def main():
    my_model = GeneralNN(10, [20, 30, 40], 50)
    print(my_model)

if __name__ == "__main__":
    main()
