from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import mido


PAD = 0
BOS = 1
EOS = 2


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Must match train.py
class CausalTransformer(torch.nn.Module):
    def __init__(self, model_cfg: Dict):
        super().__init__()
        self.cfg = model_cfg
        V = int(model_cfg["vocab_size"])
        D = int(model_cfg["d_model"])
        H = int(model_cfg["n_heads"])
        L = int(model_cfg["n_layers"])
        drop = float(model_cfg["dropout"])
        max_len = int(model_cfg["max_len"])

        self.tok = torch.nn.Embedding(V, D)
        self.pos = torch.nn.Embedding(max_len, D)

        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=D,
            nhead=H,
            dim_feedforward=4 * D,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = torch.nn.TransformerEncoder(enc_layer, num_layers=L)
        self.ln = torch.nn.LayerNorm(D)
        self.head = torch.nn.Linear(D, V, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok(x) + self.pos(pos_ids)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.tr(h, mask=causal)
        h = self.ln(h)
        return self.head(h)


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
) -> int:
    # logits: (V,)
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    if top_k and top_k > 0:
        v, idx = torch.topk(logits, k=top_k)
        probs = F.softmax(v, dim=-1)
        choice = torch.multinomial(probs, num_samples=1).item()
        return int(idx[choice].item())

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def tokens_to_midi(tokens: List[int], vocab: Dict[str, int], spec: Dict, out_path: Path) -> None:
    # Token layout from preprocess:
    time_shift_start = int(vocab["TIME_SHIFT_START"])
    note_on_start = int(vocab["NOTE_ON_START"])
    note_off_start = int(vocab["NOTE_OFF_START"])
    max_shift = int(spec["max_shift"])
    grid_ticks_per_beat = int(spec["ticks_per_beat"])

    # Output MIDI resolution
    out_ticks_per_beat = 480

    # Build absolute-time events
    t_abs = 0
    events = []  # (t_abs, type, pitch)
    for tok in tokens:
        if tok in (PAD, BOS):
            continue
        if tok == EOS:
            break

        if time_shift_start <= tok < time_shift_start + max_shift:
            shift = (tok - time_shift_start) + 1  # 1..max_shift
            delta_out = int(round(shift * (out_ticks_per_beat / grid_ticks_per_beat)))
            t_abs += max(0, delta_out)
            continue

        if note_on_start <= tok < note_on_start + 128:
            pitch = tok - note_on_start
            events.append((t_abs, "on", pitch))
            continue

        if note_off_start <= tok < note_off_start + 128:
            pitch = tok - note_off_start
            events.append((t_abs, "off", pitch))
            continue

    # Sort and convert to delta times
    events.sort(key=lambda x: x[0])

    track = mido.MidiTrack()
    mid = mido.MidiFile(ticks_per_beat=out_ticks_per_beat)
    mid.tracks.append(track)

    # tempo: 120 bpm
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))

    last_t = 0
    for t, typ, pitch in events:
        dt = max(0, t - last_t)
        last_t = t
        if typ == "on":
            track.append(mido.Message("note_on", note=int(pitch), velocity=64, time=dt))
        else:
            track.append(mido.Message("note_off", note=int(pitch), velocity=0, time=dt))

    # end of track
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/model.pt")
    ap.add_argument("--steps", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=64)
    ap.add_argument("--out", type=str, default="outputs/gen.mid")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    proc = root / "data" / "processed"
    vocab = json.loads((proc / "vocab.json").read_text())
    spec = json.loads((proc / "spec.json").read_text())

    device = pick_device()
    ckpt = torch.load(root / args.ckpt, map_location=device)

    model = CausalTransformer(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    seq = [BOS]
    max_len = int(ckpt["model_cfg"]["max_len"])

    with torch.no_grad():
        for _ in range(args.steps):
            x = torch.tensor(seq[-max_len:], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)[0, -1]  # (V,)
            nxt = sample_next(logits, temperature=args.temperature, top_k=args.top_k)
            seq.append(nxt)
            if nxt == EOS:
                break

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokens_to_midi(seq, vocab=vocab, spec=spec, out_path=out_path)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()