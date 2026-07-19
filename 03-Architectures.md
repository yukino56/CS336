# Architectures

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

![截屏2026-07-18 15.50.14](./images/%E6%88%AA%E5%B1%8F2026-07-18%2015.50.14.png)

RoPE: Rotary Position Embeddings

- 思想: word 的绝对位置不重要, "an apple" 出现在句首 / 句尾都应该是同样的 embedding

$$
\langle f(x,i),f(y,j) \rangle = g(x,y,i - j)
$$

具体的实现:

![截屏2026-07-18 16.00.24](./images/%E6%88%AA%E5%B1%8F2026-07-18%2016.00.24.png)



- we want our embeddings to be invariant to absolute position
- inner products are invariant to arbitrary rotation

高维的向量如何旋转? - 降到 2D

![截屏2026-07-18 16.03.08](./images/%E6%88%AA%E5%B1%8F2026-07-18%2016.03.08.png)

![截屏2026-07-18 16.05.41](./images/%E6%88%AA%E5%B1%8F2026-07-18%2016.05.41.png)

## Hyperparameters

### Feedforward - model dimension ratio

$$
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2
$$

Two dimensions - feedforward dim ($d_{ff}$) & model dim ($d_{model}$), 通常:
$$
d_{ff} = 4 \times d_{model}
$$
GLU 通常是:
$$
d_{ff} = \frac{8}{3}\times d_{model}
$$

### Head dim

Multi-head self-attention: 通常总维度保持不变, 即让 $d_{head} = d_{model} / n_{heads}$

### Aspect ratios

Should my model be deep or wide? How *deep* and how *wide*?

$aspect\ ratio = d_{model} / n_{layer}$, 100 is good

### Vocabulary size

token 表: Monolingual vocabs don’t need to be huge, but multilingual ones do.

### Dropout and other regularization

regularization 是防止模型过拟合的手段: 

- 比如: dropout; weight decay; label smoothing; data augmentation; early stopping
- pretrain 时是否需要 regularization?
  - 大模型预训练时数据量非常大, 不太像小数据集训练那样容易过拟合
  - 大模型预训练通常不是在同一个数据集上反复训练很多 epoch, 而是很多数据只看一遍, 所以不太会过拟合

## Stability tricks

### softmax

Softmax 存在的问题 - 大数的 exponentials / divison by zero

添加 loss 项, 让 $Z(x)$ 不要无限大:

![截屏2026-07-19 11.23.18](./images/%E6%88%AA%E5%B1%8F2026-07-19%2011.23.18.png)

### Attention softmax stability: QK norm

![截屏2026-07-19 11.26.46](./images/%E6%88%AA%E5%B1%8F2026-07-19%2011.26.46.png)

在计算 attention score 并做 softmax 之前, 先对 $Q$ 和 $K$ 做 LayerNorm 或 RMSNorm
$$
Q = LN(XW^Q)\\
K = LN(XW^K)
$$
然后计算 $scores = \frac{QK^T}{\sqrt{d}}$

### Logit soft-capping

logits: softmax 前模型会输出的一组分数 `logits = [2.1, 0.3, -1.2, 5.7]` 

![截屏2026-07-19 11.45.08](./images/%E6%88%AA%E5%B1%8F2026-07-19%2011.45.08.png)

soft-capping 是把 logits 平滑限制在范围内

## Attention heads

Multi-head self attention 通常 `num_q_heads = num_k_heads = num_v_heads` , K/V head 很多的话, KV Cache 会很大, 改造:

- GQA / MQA: 多个 Q heads 共享同一组 K/V heads
- Sparse or sliding window attention
- Exotic SSM stuff: 尝试用非 attention 结构处理长上下文

