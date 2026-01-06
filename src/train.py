from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ---------- data ----------
class JsonlTokenDataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.seqs: List[torch.Tensor] = []
        with jsonl_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                arr = json.loads(line)
                self.seqs.append(torch.tensor(arr, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.seqs[idx]


def collate_pad(batch: List[torch.Tensor], pad_id: int = 0) -> torch.Tensor:
    max_len = max(x.numel() for x in batch)
    out = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, : x.numel()] = x
    return out


# ---------- model ----------
@dataclass
class ModelCfg:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    max_len: int = 2048  # chunk length; can be >= your longest seq


class CausalTransformer(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        B, T = x.shape
        if T > self.cfg.max_len:
            raise ValueError(f"Sequence length {T} > max_len {self.cfg.max_len}")

        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok(x) + self.pos(pos_ids)

        # causal mask: True means "masked out"
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.tr(h, mask=causal)
        h = self.ln(h)
        return self.head(h)  # (B, T, V)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    proc = root / "data" / "processed"
    jsonl_path = proc / "train_chunks.jsonl"
    vocab_path = proc / "vocab.json"

    vocab = json.loads(vocab_path.read_text())
    vocab_size = int(vocab["VOCAB_SIZE"])
    pad_id = int(vocab.get("PAD", 0))

    ds = JsonlTokenDataset(jsonl_path)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_pad(b, pad_id=pad_id),
        drop_last=True,
    )

    device = pick_device()
    cfg = ModelCfg(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_len=args.max_len,
    )
    model = CausalTransformer(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.pt"

    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(dl, desc=f"epoch {epoch}/{args.epochs}")
        running = 0.0
        for step, x in enumerate(pbar, start=1):
            x = x.to(device)  # (B, T)

            # next-token prediction
            inp = x[:, :-1]
            tgt = x[:, 1:]

            logits = model(inp)  # (B, T-1, V)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=pad_id,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()
            pbar.set_postfix(loss=running / step)

        torch.save(
            {
                "model_cfg": cfg.__dict__,
                "state_dict": model.state_dict(),
                "vocab": vocab,
            },
            ckpt_path,
        )
        print(f"saved: {ckpt_path}")

    print("done")


if __name__ == "__main__":
    main()