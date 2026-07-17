# Architectures

Common architecture variations

- Activations, FFN
- Attention variants
- Position embeddings

Hyperparameters that (do or don’t) matter

- What is ff_dim? Do multi_head dims always sum to model_dim?
- How many vocab elements?

Stability tricks



## Modern variant overview

现代的网络在初始的 Transformer 修改了许多, 获得了不错的效果:

<img src="./images/%E6%88%AA%E5%B1%8F2026-07-17%2014.17.38.png" alt="截屏2026-07-17 14.17.38" style="zoom:50%;" />

(1) **LayerNorm** is in front of the block
$$
y = x + F(\text{LayerNorm}(x))
$$
(2) Rotary Position Embedding (RoPE)

(3) FFN 用 SwiGLU, 而不是 ReLU

(4) Linear layers (and layernorm) have no bias (constant) terms

## Architectures

### Pre v.s. post norm

<img src="./images/%E6%88%AA%E5%B1%8F2026-07-17%2014.30.33.png" alt="截屏2026-07-17 14.30.33" style="zoom:50%;" />

Almost all modern LMs use pre-norm (也就是右图)

<img src="./images/%E6%88%AA%E5%B1%8F2026-07-17%2015.40.00.png" alt="截屏2026-07-17 15.40.00" style="zoom:50%;" />

### LayerNorm v.s. RMSNorm

![截屏2026-07-17 15.50.05](./images/%E6%88%AA%E5%B1%8F2026-07-17%2015.50.05.png)

RMSNorm 的优点:

1. 浮点运算量小一些 (但这不是主要)
2. 更主要的是从显存读数据、写数据, 内存时间消耗大

### Drop bias terms

<img src="./images/%E6%88%AA%E5%B1%8F2026-07-17%2015.56.01.png" alt="截屏2026-07-17 15.56.01" style="zoom:40%;" />

### Activations

common activations:

![截屏2026-07-17 16.02.09](./images/%E6%88%AA%E5%B1%8F2026-07-17%2016.02.09.png)

### Gated activations (*GLU)

![截屏2026-07-17 16.06.22](./images/%E6%88%AA%E5%B1%8F2026-07-17%2016.06.22.png)

- 添加了 element-wise multiplication, 相当于 “门控”, 看保留 / 增强哪些维度

GeGLU 和 SwiGLU:

![截屏2026-07-17 16.10.26](./images/%E6%88%AA%E5%B1%8F2026-07-17%2016.10.26.png)

- 这里多了 $V$ 矩阵, 所以通常会用更小的维度 (比如降到 2/3) 来保持相似的参数规模

### Serial v.s. parallel layers

能否 parallelize the transformer block? - 改了架构, 把 attention 和 MLP 并行

![截屏2026-07-17 16.52.25](./images/%E6%88%AA%E5%B1%8F2026-07-17%2016.52.25.png)

### Position embeddings

