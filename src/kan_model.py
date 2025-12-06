import torch
import torch.nn as nn

class KAN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(KAN, self).__init__()
        
        # simple FF arch 4 KAN
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())  # replace activation function here
        
        # Remove the last [activation fuunction layer] for the output layer
        layers = layers[:-1]
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
