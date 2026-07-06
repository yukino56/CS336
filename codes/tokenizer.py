from collections import defaultdict
from sre_parse import Tokenizer

from pandas import merge

import tiktoken

def get_gpt5_tokenizer():
  # Code: https://github.com/openai/tiktoken
  return tiktoken.get_encoding("o200k_base")

def intro_to_tokenization():
  tokenizer = get_gpt5_tokenizer()
  string = "Hello, 🌍! 你好!"

  indices = tokenizer.encode(string)
  print(indices)

  reconstructed_string = tokenizer.decode(indices)
  assert string == reconstructed_string

  compression_ratio = get_compression_ratio(string, indices)
  print(f"Compression ratio: {compression_ratio:.2f}")
  print(f"Vocabulary size: {tokenizer.n_vocab}")

def get_compression_ratio(string: str, indices: list[int]) -> float:
  num_bytes = len(bytes(string, encoding="utf-8"))
  num_tokens = len(indices)
  return num_bytes / num_tokens

import dataclasses
@dataclass(frozen=True)
class BPETokenizerParams:
  vocab: dict[int, bytes]  # index -> bytes
  merges: dict[tuple[int, int], int]  # index1,index2 -> new_index

class BPETokenizer(Tokenizer):
  def __init__(self, params: BPETokenizerParams):
    self.params = params

  def encode(self, string: str) -> list[int]:
    indices = list(map(int, string.encode("utf-8")))  
    for pair, new_index in self.params.merges.items():  
        indices = merge(indices, pair, new_index)  
    return indices
  
  def decode(self, indices: list[int]) -> str:
    bytes_list = list(map(self.params.vocab.get, indices))  
    string = b"".join(bytes_list).decode("utf-8")  
    return string

def bpe_tokenizer():
  string = "the cat in the hat"
  params = train_bpe()


def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:
  # 从 bytes 开始
  indices = list(map(int, string.encode("utf-8")))  
  merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
  vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes

  for i in range(num_merges):
    counts = count_adjacent_pairs(indices)
    pair = max(counts, key=counts.get)
    new_index = 256 + i
    merges[pair] = new_index
    vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]
    indices = merge(indices, pair, new_index)

    compression_ratio = get_compression_ratio(string, indices)  
    return BPETokenizerParams(vocab=vocab, merges=merges)

def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  
  new_indices = []  
  i = 0  
  while i < len(indices):
    if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
      new_indices.append(new_index)
      i += 2
    else:
      new_indices.append(indices[i])
      i += 1
  return new_indices

def count_adjacent_pairs(indices: list[int]) -> dict[tuple[int, int], int]:
  counts = defaultdict(int)
  for index1, index2 in zip(indices, indices[1:]):
      counts[(index1, index2)] += 1
  return counts


if __name__ == "__main__":
  intro_to_tokenization()
