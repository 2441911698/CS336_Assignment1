import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
from tqdm import tqdm
PathLike = Union[str, Path]

class TextDataset(Dataset):
    """
    文本数据集：读取文本、编码为 token ID，并切分为固定长度的块。
    """
    def __init__(
        self,
        sources: Union[PathLike, Sequence[PathLike]],
        seq_len: int,
        encode: Callable[[str], List[int]],
        encoding: str = "utf-8",
    ) -> None:
        super().__init__()
        # TODO: 读取 sources 指定的文件或文件列表内容
        # TODO: 使用传入的 encode 函数将原始字符串编码为 token ID 列表
        # TODO: 检查 token 总数是否满足最低长度要求，并将其转换为 torch.Tensor 保存在类内部
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.seq_len = seq_len
        self.encode = encode

        paths:List[Path] = []
        if isinstance(sources, (str, Path)):
            paths = [Path(sources)]
        else:
            paths = [Path(p) for p in sources]

        pieces:List[str] = []
        for p in paths:
            pieces.append(p.read_text(encoding=encoding))
        text = "".join(pieces)

        ids:List[str] = []
        if len(ids) < self.seq_len + 1:
            raise ValueError(f"Need at least {self.seq_len + 1} tokens after encoding; got {len(ids)}.")
        self._data = torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        # TODO: 根据总 token 数和固定的序列长度 (self.seq_len)，计算并返回数据集可以拆分出多少个 batch 块
        return (self._data.numel() - 1) // self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: 根据索引 idx，从张量数据中截取长度为 (self.seq_len + 1) 的连续数据块
        # TODO: 执行“错位对齐” (Shift)：
        #       - 输入 x 取前 seq_len 个 token
        #       - 标签 y 取后 seq_len 个 token (即整体右移一位的预测目标)
        # TODO: 返回 (x, y) 元组
        start = idx * self.seq_len
        block = self._data[start : start + self.seq_len + 1]
        x = block[:-1].contiguous()
        y = block[1:].contiguous()
        return x, y


class Trainer:
    """
    语言模型训练器：管理单轮训练循环、验证、检查点存储与日志输出。
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: Any,
        scheduler: Any,
        train_loader: DataLoader,
        *,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        max_grad_norm: float = 1.0,
        ignore_index: int = -100,
        log_interval: int = 10,
    ) -> None:
        # TODO: 初始化并绑定传入的模型、优化器、调度器、数据加载器等组件
        # TODO: 将模型移动到指定的 device (CPU/GPU)
        # TODO: 初始化全局步数记录 (global_step) 和轮数记录 (epoch) 为 0
        self.model = model
        self.optimzer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.ignore_index = ignore_index
        self.log_interval = log_interval

        self.model.to(self.device)
        self.global_step = 0
        self.epoch = 0
        self._base_lrs: List[float] = [float(g["lr"]) for g in self.optimizer.param_groups]
        self._sched_max_lr = float(getattr(scheduler, "max_lr", max(self._base_lrs) or 1.0))

    def _sync_lr_from_scheduler(self) -> float:
        # TODO: 获取调度器 (scheduler) 在当前 global_step 下对应的学习率
        # TODO: 遍历优化器 (optimizer) 的参数组，将学习率更新进去
        # TODO: 返回当前学习率的浮点值，供日志使用
        lr = float(self.scheduler.get_lr(self.global_step))
        scale = lr / self._sched_max_lr if self._sched_max_lr > 0 else 0.0
        for g, base in zip(self.optimzer.param_groups,self._base_lrs):
            g["lr"] = base * scale
        return lr

    def _forward_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # TODO: 解析 batch 中的输入 x 和标签 y，并移动到 device 上
        # TODO: 将 x 送入模型进行前向传播，得到形状为 [B, T, V] 的预测分数 (logits)
        # TODO: 使用 .reshape(-1) 将 logits 展平为二维，将 y 展平为一维
        # TODO: 调用交叉熵损失函数计算 loss，务必传入 ignore_index 屏蔽无效填充位
        # TODO: 返回 loss 张量
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.model(x)
        if logits.dim() != 3:
            raise ValueError("Expected model(x) logits of shape (B, T, V)")
        loss = F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=self.ignore_index)
        return loss

    def train_one_epoch(self) -> Dict[str, float]:
        """训练单个 epoch 并返回统计指标。"""
        # TODO: 将模型切换为训练模式 (train)
        # TODO: 遍历 train_loader 中的每个 batch，依次执行：
        #       1. 同步当前步数的学习率
        #       2. 清空优化器梯度 (zero_grad)
        #       3. 调用 _forward_loss 计算损失
        #       4. 误差反向传播 (backward)
        #       5. 执行梯度裁剪 (clip_grad_norm_) 防止梯度爆炸
        #       6. 优化器更新参数 (step)
        # TODO: 累加处理的 token 总数和累积 loss，根据 log_interval 打印训练日志进度
        # TODO: epoch 计数器加 1，计算平均 loss 和 Perplexity 并组装成字典返回
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        t0 = time.perf_counter()

        iterator = self.train_loader
        if tqdm is not None:
            iterator = tqdm(self.train_loader, desc=f"tranin epoch {self.epoch}", leave=False)

        for batch in iterator:
            lr = self._sync_lr_from_scheduler()
            self.optimzer.zero_grad(set_to_none=True)
            loss = self._forward_loss(batch)
            loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.max_grad_norm)

            self.optimzer.step()

            bs = batch[0].size(0)
            tok = bs * batch[0].size(1)
            total_loss += float(loss.detach()) * tok
            total_tokens += tok
            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                elapsed = time.perf_counter() - t0
                tput = total_tokens / elapsed if elapsed > 0 else 0.0
                ppl = math.exp(min(float(loss.detach()), 20.0))
                msg = (
                    f"step={self.global_step} loss={float(loss):.4f} ppl={ppl:.2f} "
                    f"lr={lr:.2e} tok/s={tput:.0f}"
                )
                if tqdm is None:
                    print(msg)
                else:
                    iterator.set_postfix(loss=float(loss), lr=lr, ppl=ppl, tok_s=tput)

        self.epoch += 1
        avg_loss = total_loss / max(total_tokens, 1)
        elapsed = time.perf_counter() - t0
        out = {
            "loss": avg_loss,
            "perplexity": math.exp(min(avg_loss, 20.0)),
            "lr": self._sync_lr_from_scheduler(),
            "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
        }
        return out
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """在验证集上进行无梯度的评估。"""
        # TODO: 检查是否存在 val_loader，若无则抛出异常
        # TODO: 将模型切换为评估模式 (eval)
        # TODO: 遍历 val_loader，调用 _forward_loss 计算每个 batch 的损失
        # TODO: 累加验证集上的总 loss，返回平均 loss 和 Perplexity 指标
        if self.val_loader is None:
            raise ValueError("val_loader isnt set")
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        iterator = self.val_loader
        if tqdm is not None:
            iterator = tqdm(self.val_loader, desc="eval", leave=False)

        for batch in iterator:
            loss = self._forward_loss(batch)
            bs = batch[0].size(0)
            tok = bs * batch[0].size(1)
            total_loss += float(loss) * tok
            total_tokens += tok

        avg_loss = total_loss / max(total_tokens, 1)
        return {"loss": avg_loss, "perplexity": math.exp(min(avg_loss,20.0))}

    def save_checkpoint(self, path: PathLike, **extra: Any) -> None:
        """保存模型状态、优化器状态和训练进度。"""
        # TODO: 确保目标文件夹路径存在
        # TODO: 打包模型的 state_dict()、优化器的 state_dict()、当前的 step/epoch 计数器等信息
        # TODO: 调用 torch.save 将组装好的字典持久化保存到磁盘
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sched = {
            "max_lr": getattr(self.scheduler, "max_lr", None),
            "min_lr": getattr(self.scheduler, "min_lr", None),
            "warmup_steps": getattr(self.scheduler, "warmup_steps", None),
            "max_steps": getattr(self.scheduler, "max_steps", None),
        }
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": sched,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "base_lrs": self._base_lrs,
            "sched_max_lr": self._sched_max_lr,
            "extra": extra,
        }
        torch.save(payload, path)


    def load_checkpoint(self, path: PathLike, map_location: Optional[str] = None) -> Dict[str, Any]:
        """恢复断点训练的状态。"""
        # TODO: 使用 torch.load 读取磁盘上的检查点文件 (注意处理 device 映射 map_location)
        # TODO: 分别恢复模型的权重、优化器的状态，以及记录进度的 global_step 和 epoch
        # TODO: 返回多余的额外信息 (extra)
        path = Path(path)
        loc = map_location or str(self.device)
        ckpt = torch.load(path, map_location=loc)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = int(ckpt.get("global_step", 0))
        self.epoch = int(ckpt.get("epoch", 0))
        self._base_lrs = [float(x) for x in ckpt.get("base_lrs", self._base_lrs)]
        self._sched_max_lr = float(ckpt.get("sched_max_lr", self._sched_max_lr))
        return ckpt.get("extra", {})