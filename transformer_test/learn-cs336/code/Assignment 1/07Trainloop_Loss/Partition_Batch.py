import torch
from torch.utils.data import Dataset, DataLoader


class TokenChunkDataset(Dataset):
    """将长 token 张量切为固定长度块；标签为向右移位一位（next-token prediction）。"""
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        super().__init__()
        self.data = token_ids.long()
        self.seq_len = seq_len
        # 需要至少 seq_len+1 个 token 才能形成 seq_len 个 (x,y) 对
        self.n = max((len(self.data) - 1) // seq_len, 0)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


def collate_batch(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys