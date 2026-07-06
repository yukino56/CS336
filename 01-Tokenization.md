# Tokenization

Andrej Karpathy's video on tokenization: [video](https://www.youtube.com/watch?v=zduSFxRajkE)

## Tokenizer

Raw text is generally represented as Unicode strings. Language model takes in sequences of tokens (represented by **integers**).

$\text{string} \overset{Tokenizer}{\rightleftharpoons} \text{tokens}$

Compression Ratio = len(bytes of string) / len(tokens)

- 越大的 compression ratio, 则压缩后的序列会越短, attention 的计算量会减少
- 增大 compression ratio: 增大 vocabulary size, 比如只有字符级 token "m / a / c / ...", 那么 "machine" 要 7 个 token, 增大 vocabulary, 则可加入常见短语
- 但是会造成 sparsity, 有些 token 会不经常出现, embedding 向量可能学得不好

### Tokenizers

- Character-based tokenization: 每个 Unicode 字符对应一个 int. 比如 "a" $\rightarrow$ 97

  - Problem 1: this is a very large vocabulary.

  - Problem 2: many characters are quite rare (e.g., 🌍), which is inefficient use of the vocabulary.

- Byte-based tokenization: string 可以表示为一系列的 bytes, 可以映射到 0 到 255
  - Problem: the sequence is too long

- Word-based tokenization: split string into words
  - Problem: many words are rare and the model won't learn much about them

### Byte Pair Encoding (BPE)

思想: *train* the tokenizer on raw raw text to automatically determine the vocabulary; common sequences of characters are represented by a single token, rare sequences are represented by many tokens

- start with each byte as a token, and successively **merge** the most common pair of adjacent tokens
- 最终高频词会合并成完整的 token, 比如 the; 低频词会被拆成片段, 比如 unbelievable 可能被拆成 `un` + `believ` + `able`
