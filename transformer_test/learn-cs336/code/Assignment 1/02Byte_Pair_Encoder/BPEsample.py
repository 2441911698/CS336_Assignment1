"""
BPE 教学实现 — CS336 Lesson 02
"""
from __future__ import annotations

import multiprocessing as mp
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import regex as re

GPT2_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize_gpt2(text: str) -> List[str]:
    return [m.group(0) for m in GPT2_SPLIT_PATTERN.finditer(text)]


def bytes_to_ids(chunk: str) -> List[int]:
    return list(chunk.encode("utf-8"))


def get_stats(ids: Sequence[int], counts: Optional[Dict[Tuple[int, int], int]] = None) -> Dict[Tuple[int, int], int]:
    counts = counts if counts is not None else {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    a, b = pair
    out: List[int] = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def count_pairs_for_chunks(chunk_freqs: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, int], int]:
    stats: Dict[Tuple[int, int], int] = {}
    for chunk_tuple, freq in chunk_freqs.items():
        #freq 单词频率
        ids = list(chunk_tuple)
        for i in range(len(ids) - 1):
            p = (ids[i], ids[i + 1])
            stats[p] = stats.get(p, 0) + freq
    return stats  #字节对频率


def train_bpe(text_corpus: str, num_merges: int, pattern: re.Pattern = GPT2_SPLIT_PATTERN):
    chunk_freqs: Dict[Tuple[int, ...], int] = Counter()
    for m in pattern.finditer(text_corpus):
        t = tuple(bytes_to_ids(m.group(0)))
        chunk_freqs[t] += 1

    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: List[Tuple[int, int, int]] = []
    next_id = 256

    for _ in range(num_merges):
        pair_stats = count_pairs_for_chunks(chunk_freqs)
        if not pair_stats:
            break
        best_pair = max(pair_stats.items(), key=lambda kv: (kv[1], kv[0]))[0]
        left, right = best_pair
        new_chunk_freqs: Dict[Tuple[int, ...], int] = defaultdict(int)
        for chunk_tuple, freq in chunk_freqs.items():
            merged = merge(list(chunk_tuple), best_pair, next_id)
            new_chunk_freqs[tuple(merged)] += freq
        chunk_freqs = dict(new_chunk_freqs)
        vocab[next_id] = vocab[left] + vocab[right]
        merges.append((left, right, next_id))
        next_id += 1

    return vocab, merges


def build_merge_ranks(merges: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], int]:
    return {(a, b): r for r, (a, b, _) in enumerate(merges)}


def encode_piece_by_rank(ids: List[int], merges: List[Tuple[int, int, int]]) -> List[int]:
    pair_to_new = {(a, b): nid for a, b, nid in merges}
    merge_ranks = build_merge_ranks(merges)
    seq = ids[:]
    while True:
        best_rank, pos = None, None
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i + 1])
            if p not in merge_ranks:
                continue
            r = merge_ranks[p]
            if best_rank is None or r < best_rank or (r == best_rank and pos is not None and i < pos):
                best_rank, pos = r, i
        if best_rank is None or pos is None:
            break
        p = (seq[pos], seq[pos + 1])
        new_id = pair_to_new[p]
        seq = seq[:pos] + [new_id] + seq[pos + 2 :]
    return seq


def encode_piece_sequential(ids: List[int], merges: List[Tuple[int, int, int]]) -> List[int]:
    seq = ids[:]
    for left, right, new_id in merges:
        seq = merge(seq, (left, right), new_id)
    return seq


def bpe_encode(text: str, merges: List[Tuple[int, int, int]], pattern: re.Pattern = GPT2_SPLIT_PATTERN) -> List[int]:
    out: List[int] = []
    for piece in pretokenize_gpt2(text):
        out.extend(encode_piece_sequential(bytes_to_ids(piece), merges))
    return out


def bpe_decode(ids: Sequence[int], vocab: Dict[int, bytes]) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def worker_count_pairs(lines: List[str], pattern: re.Pattern) -> Dict[Tuple[int, int], int]:
    local: Dict[Tuple[int, int], int] = {}
    for line in lines:
        for m in pattern.finditer(line):
            ids = bytes_to_ids(m.group(0))
            for i in range(len(ids) - 1):
                p = (ids[i], ids[i + 1])
                local[p] = local.get(p, 0) + 1
    return local


def parallel_count_pairs(
    corpus_lines: List[str], num_workers: int = 4, pattern: re.Pattern = GPT2_SPLIT_PATTERN
) -> Dict[Tuple[int, int], int]:
    if num_workers <= 1:
        return worker_count_pairs(corpus_lines, pattern)
    chunk_size = max(1, len(corpus_lines) // num_workers)
    shards = [corpus_lines[i : i + chunk_size] for i in range(0, len(corpus_lines), chunk_size)]
    with mp.Pool(num_workers) as pool:
        parts = pool.starmap(worker_count_pairs, [(s, pattern) for s in shards])
    total: Dict[Tuple[int, int], int] = {}
    for d in parts:
        for k, v in d.items():
            total[k] = total.get(k, 0) + v
    return total


if __name__ == "__main__":
    sample = "hello hello hello 你好"
    vocab, merges = train_bpe(sample, num_merges=10)
    ids = bpe_encode(sample, merges)
    assert bpe_decode(ids, vocab) == sample
    print("token ids:", ids)
    print("OK")