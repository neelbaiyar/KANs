import torch
import torch.nn as nn
import torch.nn.functional as F
from kan_model import KAN
import math

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Sawtooth(nn.Module):
    def forward(self, x):
        return x - torch.floor(x)

class SinLinear(nn.Module):
    def forward(self, x):
        return x * torch.sin(x)

class Abs(nn.Module):
    def forward(self, x):
      return torch.abs(x)

class SoftAbs(nn.Module):
    def forward(self, x):
      return x**2 / (1 + torch.abs(x))

class Sine(nn.Module):
    def forward(self, x):
      return torch.sin(x)

class Sinc(nn.Module):
    def forward(self, x):
      return torch.sin(x) / x

class Sinh(nn.Module):
    def forward(self, x):
      return torch.sinh(x)

class Nand(nn.Module):
    def forward(self, x):
      return torch.clamp(2 - x, min=0, max=1)

class Nor(nn.Module):
    def forward(self, x):
      return torch.clamp(x, 0, 1)

class Xor(nn.Module):
    def forward(self, x):
      return 1 - torch.abs(0.5*x - torch.floor(0.5*x) - 0.5)*2

class SmoothXor(nn.Module):
    def forward(self, x):
      x2 = math.pi*(x - 0.75)
      h = torch.sin(2*x2) - torch.sin(6*x2)/9 + torch.sin(10*x2)/25
      return 0.5 + h / 2.30222

class ClippedCubicSigmoid(nn.Module):
    def forward(self, x):
      return torch.where(
        x < -1,
        0,
        torch.where(
          x > 1,
          1,
          0.5 - (x**3)/4 + 0.75*x
        )
      )

class SquareWave(nn.Module):
    def forward(self, x):
      return 0.5*torch.sign(torch.sin(x)) + 0.5

class SawtoothRamp(nn.Module):
    def forward(self, x):
      return (13*x - torch.floor(10*x)) / 5

class Perlin1D(nn.Module):
    # fade function: 6t^5 - 15t^4 + 10t^3
    def fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    # linear interpolation
    def lerp(self, a, b, t):
        return a + t * (b - a)

    # hash -> deterministic pseudo-random gradients in [-1, 1]
    def grad(self, x):
        # Use a PyTorch-safe hash and mapping to [-1, 1]
        h = (x * 15731 + 789221) * x + 1376312589
        return torch.frac(torch.sin(h.float()) * 43758.5453) * 2 - 1

    def forward(self, x):
        x = x * 7 # increase spatial frequency

        # 1. grid points
        x0 = torch.floor(x)
        x1 = x0 + 1

        # 2. distances
        t = x - x0

        # 3. gradients
        g0 = self.grad(x0.int())
        g1 = self.grad(x1.int())

        # 4. dot products
        d0 = t
        d1 = t - 1

        v0 = g0 * d0
        v1 = g1 * d1

        # 5. interpolate
        u = self.fade(t)
        return self.lerp(v0, v1, u)

class Weierstrass(nn.Module):
    def forward(self, x):
      W = 0
      for n in range(7):
        W = W + 0.5**n * torch.cos(3**n * 3.14 * x)
      return W

class CantorFunction(nn.Module):
    def forward(self, x):
      result = torch.zeros_like(x)
      mask = torch.ones_like(x, dtype=torch.bool) # tracks active elements
      for i in range(7): # going with a depth of 7
        x3 = x*3
        digit = torch.floor(x3).to(torch.int64)
        x = x3 - digit

        add_mask = mask & (digit == 2)
        result[add_mask] += 1 / 2**i
        mask = mask & (digit != 1)
      return result

class SineLU(nn.Module):
    def forward(self, x):
        return torch.sin(x) * F.relu(x)

ACTIVATION_FACTORIES = {
    "relu":      lambda: nn.ReLU(),
    "leakyrelu": lambda: nn.LeakyReLU(0.1),
    "gelu":      lambda: nn.GELU(),
    "silu":      lambda: nn.SiLU(),
    "tanh":      lambda: nn.Tanh(),
    "sigmoid":   lambda: nn.Sigmoid(),
    "softplus":  lambda: nn.Softplus(),
    "mish":      lambda: Mish(),
    "sawtooth":  lambda: Sawtooth(),
    "sinlinear": lambda: SinLinear(),
    "identity":  lambda: nn.Identity(),
    "abs":       lambda: Abs(),
    "softabs":   lambda: SoftAbs(),
    "sine":      lambda: Sine(),
    "sinc":      lambda: Sinc(),
    "sinh":      lambda: Sinh(),
    "sinelu":    lambda: SineLU(),
    "nand":      lambda: Nand(),
    "nor":       lambda: Nor(),
    "xor":       lambda: Xor(),
    "smoothxor": lambda: SmoothXor(),
    "cubicsigm": lambda: ClippedCubicSigmoid(),
    "squarewav": lambda: SquareWave(),
    "sawtoothramp": lambda: SawtoothRamp(),
    "perlin":    lambda: Perlin1D(),
    "weierstrass": lambda: Weierstrass(),
    "cantor":    lambda: CantorFunction()
}
