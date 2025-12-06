import torch.nn as nn
from activations import ACTIVATION_FACTORIES

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_name: str):
        super().__init__()
        if activation_name not in ACTIVATION_FACTORIES:
            raise ValueError(f"Unknown activation: {activation_name}")
        act_factory = ACTIVATION_FACTORIES[activation_name]

        layers = []
        dims = [input_dim] + hidden_dims
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(act_factory())

        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)
        self.activation_name = activation_name

    def forward(self, x):
        return self.net(x)