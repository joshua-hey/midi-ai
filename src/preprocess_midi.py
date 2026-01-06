from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json

import mido
from tqdm import tqdm


# ---------- Token spec ----------
# TIME_SHIFT: 1..MAX_SHIFT ticks (ticks are your quantization grid, e.g., 16th note)
# NOTE_ON: pitch 0..127
# NOTE_OFF: pitch 0..127
#
# Token IDs are contiguous:
# 0                          -> PAD
# 1                          -> BOS
# 2                          -> EOS
# 3..(3+MAX_SHIFT-1)         -> TIME_SHIFT(1..MAX_SHIFT)
# (after that) 128 tokens    -> NOTE_ON(0..127)
# (after that) 128 tokens    -> NOTE_OFF(0..127)

PAD = 0
BOS = 1
EOS = 2


@dataclass(frozen=True)
class TokenSpec:
    ticks_per_beat: int = 4          # 4 ticks/beat => 16th-note grid in 4/4
    max_shift: int = 16              # max 4 beats if ticks_per_beat=4
    min_pitch: int = 21              # piano A0
    max_pitch: int = 108             # piano C8


def build_vocab(spec: TokenSpec) -> Dict[str, int]:
    base = 3
    time_shift_start = base
    note_on_start = time_shift_start + spec.max_shift
    note_off_start = note_on_start + 128
    vocab = {
        "PAD": PAD,
        "BOS": BOS,
        "EOS": EOS,
        "TIME_SHIFT_START": time_shift_start,
        "NOTE_ON_START": note_on_start,
        "NOTE_OFF_START": note_off_start,
        "VOCAB_SIZE": note_off_start + 128,
    }
    return vocab


def token_time_shift(n: int, vocab: Dict[str, int]) -> int:
    return vocab["TIME_SHIFT_START"] + (n - 1)


def token_note_on(pitch: int, vocab: Dict[str, int]) -> int:
    return vocab["NOTE_ON_START"] + pitch


def token_note_off(pitch: int, vocab: Dict[str, int]) -> int:
    return vocab["NOTE_OFF_START"] + pitch


# ---------- MIDI utilities ----------
def read_midi(path: Path) -> mido.MidiFile:
    return mido.MidiFile(str(path))


def merge_tracks_to_messages(mid: mido.MidiFile) -> List[mido.Message]:
    # mido.merge_tracks keeps delta-times (in ticks) on returned messages
    return list(mido.merge_tracks(mid.tracks))


def filter_to_piano(msgs: List[mido.Message]) -> List[Tuple[int, mido.Message]]:
    """
    Returns (delta_ticks, message) only for note_on/note_off.
    If your MIDI has multiple instruments, this ignores program changes and other CC.
    """
    out: List[Tuple[int, mido.Message]] = []
    for m in msgs:
        if m.is_meta:
            continue
        if m.type in ("note_on", "note_off"):
            out.append((m.time, m))
    return out


def normalize_note_off(m: mido.Message) -> Tuple[str, int, int]:
    """
    Returns (event_type, pitch, velocity).
    Treat note_on with velocity=0 as note_off.
    """
    if m.type == "note_on" and m.velocity == 0:
        return ("note_off", m.note, 0)
    if m.type == "note_on":
        return ("note_on", m.note, m.velocity)
    return ("note_off", m.note, m.velocity)


def quantize_ticks(delta_ticks: int, mid_ticks_per_beat: int, spec: TokenSpec) -> int:
    """
    Convert MIDI delta ticks to our grid ticks.
    spec.ticks_per_beat defines the grid resolution.
    """
    if delta_ticks <= 0:
        return 0
    # Convert to beats, then to grid ticks, then round to nearest int
    beats = delta_ticks / float(mid_ticks_per_beat)
    grid_ticks = round(beats * spec.ticks_per_beat)
    return int(max(0, grid_ticks))


def encode_midi_to_tokens(path: Path, spec: TokenSpec, vocab: Dict[str, int]) -> List[int]:
    mid = read_midi(path)
    msgs = merge_tracks_to_messages(mid)
    msgs = filter_to_piano(msgs)

    tokens: List[int] = [BOS]
    for delta_ticks, m in msgs:
        q = quantize_ticks(delta_ticks, mid.ticks_per_beat, spec)
        # Emit time shift tokens
        while q > 0:
            step = min(q, spec.max_shift)
            tokens.append(token_time_shift(step, vocab))
            q -= step

        ev, pitch, _vel = normalize_note_off(m)

        # pitch range clamp/skip
        if pitch < spec.min_pitch or pitch > spec.max_pitch:
            continue

        if ev == "note_on":
            tokens.append(token_note_on(pitch, vocab))
        else:
            tokens.append(token_note_off(pitch, vocab))

    tokens.append(EOS)
    return tokens


def chunk_tokens(tokens: List[int], chunk_len: int) -> List[List[int]]:
    """
    Create fixed-length chunks for training.
    Keeps BOS/EOS inside chunks if they fall there.
    """
    if len(tokens) <= chunk_len:
        return [tokens]
    chunks: List[List[int]] = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_len]
        if len(chunk) < 8:
            break
        chunks.append(chunk)
        i += chunk_len
    return chunks


def main() -> None:
    spec = TokenSpec()
    vocab = build_vocab(spec)

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw_midi"
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_paths = sorted([p for p in raw_dir.rglob("*.mid")] + [p for p in raw_dir.rglob("*.midi")])
    if not midi_paths:
        raise SystemExit(f"No MIDI files found in: {raw_dir}")

    chunk_len = 2048
    all_chunks: List[List[int]] = []
    bad: List[str] = []

    for p in tqdm(midi_paths, desc="Encoding MIDIs"):
        try:
            toks = encode_midi_to_tokens(p, spec, vocab)
            chunks = chunk_tokens(toks, chunk_len=chunk_len)
            all_chunks.extend(chunks)
        except Exception as e:
            bad.append(f"{p}: {type(e).__name__}: {e}")

    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2))
    (out_dir / "spec.json").write_text(json.dumps(spec.__dict__, indent=2))
    (out_dir / "bad_files.txt").write_text("\n".join(bad))

    # Save chunks as JSONL: one chunk per line
    out_path = out_dir / "train_chunks.jsonl"
    with out_path.open("w") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch) + "\n")

    print(f"Found MIDIs: {len(midi_paths)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Wrote: {out_path}")
    if bad:
        print(f"Bad files: {len(bad)} (see {out_dir / 'bad_files.txt'})")


if __name__ == "__main__":
    main()