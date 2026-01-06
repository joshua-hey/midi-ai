# MIDI AI Generator

Minimal end-to-end Transformer for generating piano MIDI music.

This project:
- Tokenizes MIDI files into discrete events
- Trains a causal Transformer on those tokens
- Generates new MIDI files from the trained model

Local, simple, reproducible.

---

## Requirements

- Python 3.11
- Git

Dependencies:
```
torch
numpy
mido
tqdm
```

---

## Structure

```
midi-ai/
├── src/
│   ├── preprocess_midi.py
│   ├── train.py
│   └── generate.py
├── data/
│   ├── raw_midi/
│   └── processed/
├── checkpoints/
├── outputs/
├── requirements.txt
└── README.md
```

---

## Setup

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data

Put MIDI files in:
```
data/raw_midi/
```

---

## Preprocess

```
python -m src.preprocess_midi
```

Creates tokenized training data in `data/processed/`.

---

## Train

```
python -m src.train --epochs 1 --batch 8
```

Writes `checkpoints/model.pt`.

---

## Generate

```
python -m src.generate --steps 1500 --temperature 1.0 --top_k 64
```

Output:
```
outputs/gen.mid
```

Open in any MIDI player or DAW.

---

## Notes

- Piano-only, note events only
- No audio modeling
- Minimal baseline for experimentation