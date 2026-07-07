# PyTorch, resource accounting

Transformer:

- [Assignment 1 handout](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2025_assignment1_basics.pdf)
- [Mathematical description](https://johnthickstun.com/docs/transformers.pdf)
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

## Memory accounting

> https://docs.pytorch.org/docs/stable/tensors.html

### Tensors

```python
x = torch.zeros(4, 8)
x = torch.ones(4, 8)
x = torch.randn(4, 8)	# 4x8 matrix of iid Normal(0, 1)
```

In AI world, almost everything are stored as **floating point numbers**.

float32:

<img src="https://stanford-cs336.github.io/spring2025-lectures/images/fp32.png">

float32 / fp16 / bf16 要 trade off, 用 **Mixed precision training**:

- Use bf16 for parameters, activations, and gradients
- Use fp32 for optimizer states

### Tensors on GPUs

By default, tensors are stored in CPU memory.

```python
x = torch.zeros(32, 32)
assert x.device == torch.device("cpu")
```

We need to explicitly move tensors to **GPU**

<img src="https://stanford-cs336.github.io/spring2025-lectures/images/cpu-gpu.png" width="60%">

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 把模型和数据放到 device 上
model = model.to(device)
x = x.to(device)
y = y.to(device)
```

Mac 没有 CUDA, 可以用 Apple 的 MPS

```python
if torch.cuda.is_available():
  device = torch.device("cuda")
elif torch.backends.mps.is_available():
  device = torch.device("mps")
else:
  device = torch.device("cpu")
print(device)
```

## Compute accounting

### einops

> [Einops tutorial](https://einops.rocks/1-einops-basics/)

作用: 给每个维度起名字, 让矩阵运算更清楚

<u>einsum:</u>

```python
x = torch.ones(3, 4)    # seq1 x hidden
y = torch.ones(4, 3)    # hidden x seq2
z = x @ y

z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")
```

更复杂的例子:

```python
x = torch.ones(2, 3, 4)   # batch seq1 hidden
y = torch.ones(2, 3, 4)   # batch seq2 hidden
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
```

> **Dimensions that are not named in the output are summed over; any dimension that is named is just iterated over.**

<u>reduce:</u> reduce a single tensor via some operation (e.g., sum, mean, max, min)

```python
x = torch.ones(2, 3, 4)  # batch seq hidden
y = reduce(x, "... hidden -> ...", "sum")
```

<u>rearrange:</u> 

```python
x = torch.ones(3, 8)  # seq total_hidden
w = torch.ones(4, 4)  # hidden1 hidden2
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
```

### FLOP

floating-point operations (FLOP)

- FLOPs: floating-point operations (measure of computation done)
- FLOP/s: floating-point operations per second

B x D 和 D x K 阶矩阵相乘, 总的 FLOP 数为 2BDK. Matrix multiplications dominate.

- 如果将 B 视作向量数, D 为维度, D x K 是权重矩阵, 那么 $\text{FLOPs} = 2 \times \text{(\# tokens)} \times \text{\# parameters}$

### arithmetic intensity

<img src="https://github.com/stanford-cs336/lectures/blob/main/images/compute-memory.png?raw=true" width="40%">

时间消耗: 1) 在 memory 和 gpu 传数据; 2) 计算

- Accelerator speed (FLOP/s)
- Memory bandwidth (bytes/s)

假设能使 communication 和 computation overlap

- 会发现基本上都是 memory bound (比如 ReLU), 即传输花的时间长
- 矩阵乘法是 compute bound!

## Memory and compute accounting for *Training*

### Deep Network

<img src="https://github.com/stanford-cs336/lectures/blob/main/images/deep-network.png?raw=true">

- $L$ layers and $D$-dimensional inputs, activations and outputs

```python
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
```

### Gradients

> 前面是在 forward, 现在我们来 backward

E.g. linear model: $y = xw, l = \frac{1}{2}(xw - 5)^2$

```python
x = torch.tensor([1., 2, 3])
w = torch.tensor([1., 1, 1], requires_grad=True)
pred_y = x @ w
loss = 0.5 * (pred_y - 5).pow(2)
loss.backward()
assert torch.equal(w.grad, torch.tensor([1, 2, 3]))
```

- `loss.backward()` : $\frac{\partial l}{\partial w} = \frac{\partial l}{\partial y}\times \frac{\partial y}{\partial w} = (y - 5)x$

### FLOPs

> 计算 gradients 的时候的 FLOPs

E.g. Model: x --w1--> h1 --w2--> h2 -> loss

```python
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
```

h2 = h1 @ w2:

- Forward pass: [B, D] @ [D, D] -> [B, D] 每个输出元素要 D 次乘法 D 次加法, 有 B x D 个元素, 所以是 2 B x D x D
- Backward pass: 
  - h1.grad = d loss / d h1 = h2.grad @ w2.T
  - w2.grad = d loss / d w2 =  h1.T @ h2.grad
  - 一共是 4 B x D x D

Total: **6** (# data points) (# parameters) FLOPs

## Optimizer

> https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

Define the AdaGrad optimizer

- momentum = SGD + exponential averaging of grad
- AdaGrad = SGD + averaging by grad^2
- RMSProp = AdaGrad + exponentially averaging of grad^2
- Adam = RMSProp + momentum





### operations

PyTorch tensors are **<u>pointers</u>** into allocated memory.

<img src="https://stanford-cs336.github.io/spring2025-lectures/var/files/image-97aa05a6701b46521cb8a7c1e096c7e7-https_martinlwx_github_io_img_2D_tensor_strides_png">

```python
>>> x = torch.tensor([
    [0., 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
])

>>> x.stride()
(4, 1)
```

To go to the next row (dim 0), skip 4 elements in storage:

```python
assert x.stride(0) == 4
```

To go to the next column (dim 1), skip 1 element in storage:

```python
assert x.stride(1) == 1
```

To find an element:

```python
r, c = 1, 2
index = r * x.stride(0) + c * x.stride(1)
assert index == 6
```

#### View

Many operations simply provide a different view of the tensor.

```python
x = torch.tensor([[1., 2, 3], [4, 5, 6]])

y = x[0]	# get row 0
assert same_storage(x, y)

y = x[:, 1]	# get column 1
assert torch.equal(y, torch.tensor([2, 5]))
assert same_storage(x, y)

y = x.view(3, 2)	# view 2x3 matrix as 3x2 matrix (本质上都是线性排列的)
assert torch.equal(y, torch.tensor([[1, 2], [3, 4], [5, 6]]))
assert same_storage(x, y)

y = x.transpose(1, 0)
assert torch.equal(y, torch.tensor([[1, 4], [2, 5], [3, 6]]))
assert same_storage(x, y)

x[0][0] = 100	# mutating x also mutates y
assert y[0][0] == 100
```

Some views are non-contiguous entries, which means that further views aren't possible

```python
x = torch.tensor([[1., 2, 3], [4, 5, 6]])
y = x.transpose(1, 0)
assert not y.is_contiguous()
```

- 内存并未改变, 是 stride 变了

Enforce a tensor to be contiguous:

```python
y = x.transpose(1, 0).contiguous().view(2, 3)
assert not same_storage(x, y)
```

- 这会进行内存的拷贝与重新排列

#### matmul

```python
x = torch.ones(16, 32)
w = torch.ones(32, 2)
y = x @ w
assert y.size() == torch.Size([16, 2])
```

## Models

### parameters

Model parameters are stored in PyTorch as `nn.Parameter` objects.

```python
import torch.nn as nn
w = nn.Parameter(torch.randn(input_dim, output_dim))
assert isinstance(w, torch.Tensor)  # Behaves like a tensor
assert type(w.data) == torch.Tensor  # Access the underlying tensor
```

Parameter initialization: Large values can cause gradients to blow up and cause training to be unstable. We want an initialization that is invariant to `input_dim`. We simply rescale by 1/sqrt(input_dim)

```python
import numpy as np
w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
output = x @ w
```

To be extra safe, we truncate the normal distribution to [-3, 3] to avoid any chance of outliers.

```python
w = nn.Parameter(nn.init.trunc_normal_(torch.empty(input_dim, output_dim), std=1 / np.sqrt(input_dim), a=-3, b=3))
```

### randomness

For reproducibility, it is recommended that we always pass in a different random seed for each use of randomness. There are three places to set the random seed which we should do all at once just to be safe.

```python
# Torch
seed = 0
torch.manual_seed(seed)

# NumPy
np.random.seed(seed)

# Python
random.seed(seed)
```

### data loading

如果不想把所有数据装入内存, 使用 memmap:

```python
data = np.memmap("data.npy", dtype=np.int32)
```

## Transformer

> https://www.adamcasson.com/posts/transformer-flops
