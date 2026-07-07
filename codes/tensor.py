import math
import torch
from einops import einsum
from einops import reduce
from einops import rearrange
from torch import nn

def tensor_basic():
  x = torch.zeros(4)
  print(x)
  x = torch.zeros(4, 8)
  print(x)
  x = torch.ones(4, 8)
  print(x)
  x = torch.zeros(4, 8, 2)
  print(x)
  x = torch.randn(4, 8)   # 4x8 matrix of iid Normal(0, 1)
  print(x)

  if torch.cuda.is_available():
    device = torch.device("cuda")
  elif torch.backends.mps.is_available():
    device = torch.device("mps")
  else:
    device = torch.device("cpu")
  print(device)

def einops():
  x = torch.ones(3, 4)    # seq1 x hidden
  y = torch.ones(4, 3)    # hidden x seq2
  z = x @ y

  z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")
  print(z)

  x = torch.ones(2, 3, 4)   # batch seq1 hidden
  y = torch.ones(2, 3, 4)   # batch seq2 hidden
  z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
  print(z)

  x = torch.ones(2, 3, 4)  # batch seq hidden
  y = reduce(x, "... hidden -> ...", "sum")
  print(y)

  x = torch.ones(3, 8)  # seq total_hidden
  w = torch.ones(4, 4)  # hidden1 hidden2
  x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
  print(x)

## Below is a simple deep network

import torch.nn.functional as F

def deep_network():
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
  D = 8
  L = 3
  model = DeepNetwork(dim=D, num_layers=L).to(device)

  num_parameters = get_num_parameters(model)
  assert num_parameters == (D * D) * L

  B = 4   # batch size
  x = torch.randn(B, D).to(device)
  y = model(x)  # 会调用 forward() 函数
  print(y)

def get_num_parameters(model: nn.Module) -> int:
  return sum(param.numel() for param in model.parameters())

class Block(nn.Module):
  """
  Simple block that applies a linear transformation followed by a ReLU non-linearity.
  """
  def __init__(self, dim: int):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x @ self.weight   # Linear
    x = F.relu(x)         # Activation
    return x

class DeepNetwork(nn.Module):
  def __init__(self, dim: int, num_layers: int):
    super().__init__()
    self.layers = nn.ModuleList([Block(dim) for _ in range(num_layers)])
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      x = layer(x)
    return x

def gradients():
  x = torch.tensor([1., 2, 3])
  w = torch.tensor([1., 1, 1], requires_grad=True)
  pred_y = x @ w
  loss = 0.5 * (pred_y - 5).pow(2)
  loss.backward()
  assert torch.equal(w.grad, torch.tensor([1, 2, 3]))

def gradients_flops():
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

  B = 1024
  D = 256

  x = torch.ones(B, D).to(device)
  w1 = torch.ones(D, D, requires_grad=True).to(device)
  w2 = torch.ones(D, D, requires_grad=True).to(device)

  # forward pass
  h1 = einsum(x, w1, "batch in, in out -> batch out")
  h2 = einsum(h1, w2, "batch in, in out -> batch out")
  loss = (h2.mean() - 0) ** 2

  # backward pass
  h1.retain_grad()  # debugging
  h2.retain_grad()
  loss.backward()

  h1_grad = einsum(h2.grad, w2, "batch out, int out -> batch in")
  w2_grad = einsum(h2.grad, h1, "batch out, batch in -> in out")

if __name__ == "__main__":
  # tensor_basic()
  # einops()
  # deep_network()
  # gradients()
  gradients_flops()