import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Sawtooth(nn.Module):
    def forward(self, x):
        return x - torch.floor(x)

class SinLinear(nn.Module):
    def forward(self, x):
        return x * torch.sin(x)

ACTIVATION_FACTORIES = {
    "relu":      lambda: nn.ReLU(),
    "leakyrelu": lambda: nn.LeakyReLU(0.1),
    "gelu":      lambda: nn.GELU(),
    "silu":      lambda: nn.SiLU(),      # Swish
    "tanh":      lambda: nn.Tanh(),
    "sigmoid":   lambda: nn.Sigmoid(),
    "softplus":  lambda: nn.Softplus(),
    "mish":      lambda: Mish(),
    "sawtooth":  lambda: Sawtooth(),
    "sinlinear": lambda: SinLinear(),
}
