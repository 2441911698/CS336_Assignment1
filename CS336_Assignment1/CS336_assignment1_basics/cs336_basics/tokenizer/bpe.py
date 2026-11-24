from __future__ import annotations

import json
import re
from collections import Counter
from typing import Iterable

try:
    import regex

    _GPT2_SPLIT_RE = regex.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:  # pragma: no cover - fallback for environments without `regex`
    _GPT2_SPLIT_RE = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\d+| ?[^\s]+|\s+(?!\S)|\s+"""
    )


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    if not ids:
        return []
    new_ids: list[int] = []
    a, b = pair
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def _pretokenize(text: str) -> list[str]:
    if not text:
        return []
    return [m.group(0) for m in _GPT2_SPLIT_RE.finditer(text)]


def _build_vocab_from_merges(merges: list[tuple[int, int]]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for a, b in merges:
        vocab[next_id] = vocab[a] + vocab[b]
        next_id += 1
    return vocab


class BPETokenizer:

    def __init__(self, vocab_size: int = 512) -> None:
        self._target_vocab_size = vocab_size
        self.merges: list[tuple[int, int]] = []
        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab(self) -> dict[int, bytes]:
        return dict(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def train(self, text: str, vocab_size: int | None = None) -> None:
        target = vocab_size if vocab_size is not None else self._target_vocab_size
        if target < 256:
            raise ValueError("vocab_size must be >= 256")

        num_merges = target - 256
        if num_merges == 0:
            self.merges = []
            self._vocab = {i: bytes([i]) for i in range(256)}# 清零词表
            return

        # 1. 预切分文本
        chunks = _pretokenize(text)
        if not chunks:
            return

        # 2. 将每个片段转为初始的 ID 列表 (0-255 的整数)
        seqs: list[list[int]] = [list(ch.encode("utf-8")) for ch in chunks]

        self.merges = []
        self._vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        for _ in range(num_merges):
            pair_counts:Counter[tuple[int,int]] = Counter() # 统计当前seqs中所有相邻token对的频次
            for seq in seqs:
                pair_counts.update(get_stats(seq))
            if not pair_counts:
                break # 提前防错退出机制
            best_count = max(pair_counts.values()) # 找出频次最高的 pair。
            candidates = [p for p,c in pair_counts.items() if c == best_count] # 用min打破平局
            pair = min(candidates)

            seqs = [merge(seq,pair,next_id)for seq in seqs] # 遍历seqs，调用merge函数，用next_id替换最佳pair
            self.merges.append(pair)
            self._vocab[next_id] = self._vocab[pair[0]] + self._vocab[pair[1]] # 更新self.merges列表和self._vocab字典
            next_id += 1

        self._target_vocab_size = target

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        out: list[int] = []
        for chunk in _pretokenize(text):
            seq = list (chunk.encode("utf-8"))
            for idx, pair in enumerate(self.merges):
                seq = merge(seq,pair,256+idx)
            out.extend(seq)
        return out

    def decode(self, ids: Iterable[int]) -> str:
        """
        Decode token IDs back to a UTF-8 string.
        将 token ID 序列解码为 UTF-8 字符串。
        """
        data = bytearray()
        for id in ids:
            if id not in self._vocab:
                raise KeyError(f"Unknown token id: {id}")
            else:
                data.extend(self._vocab[id])
        return bytes(data).decode("utf-8",errors = "replace")
        # ==========================================
        # TODO: 实现解码逻辑
        # ==========================================
        # 1. 遍历 ids 里的每个 token id
        # 2. 检查它是否在 self._vocab 中，不在则抛出 KeyError
        # 3. 如果在，将其对应的 bytes 追加到 data 中
        # 4. 最后将 data 解码为 utf-8 字符串 (建议加上 errors="replace" 防止遇到无效字节)


    # ---------------------------------------------------------
    # IO 方法已为你保留完整实现，以便直接测试核心算法
    # ---------------------------------------------------------
    def save(self, path: str) -> None:
        """Save merges and metadata to JSON."""
        payload = {
            "model": "bpe_byte_gpt2pretok",
            "merges": [list(p) for p in self.merges],
            "target_vocab_size": self._target_vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Load tokenizer state from a file written by `save`."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        merges_raw = payload.get("merges", [])
        self.merges = [(int(a), int(b)) for a, b in merges_raw]
        self._target_vocab_size = int(payload.get("target_vocab_size", 256 + len(self.merges)))
        # 这里需要用到你之前写的 _build_vocab_from_merges
        self._vocab = _build_vocab_from_merges(self.merges)

    @classmethod
    def from_file(cls, path: str) -> "BPETokenizer":
        tok = cls()
        tok.load(path)
        return tok


if __name__ == "__main__":
    sample = (
        "Hello, world! 你好，世界！\n"
        "BPE learns frequent pairs like 'th' and multi-byte UTF-8 sequences.\n"
        "The tokenizer uses GPT-2 style regex pretokenization."
    )
    print("Training BPE on sample text... / 在示例文本上训练 BPE...\n")
    tok = BPETokenizer(vocab_size=320)
    tok.train(sample, vocab_size=320)

    print(f"vocab_size = {tok.vocab_size}, merges = {len(tok.merges)}")
    ids = tok.encode(sample)
    back = tok.decode(ids)

    assert back == sample, "Round-trip should preserve text / 往返应无损"
    print(f"encode length: {len(ids)} tokens")
    print(f"decode matches: {back == sample}")
    print("Demo OK. / 演示通过。")


