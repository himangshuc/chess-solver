# Chess Solver — Image → Best Move

Upload or snap a photo of a chessboard. The app detects the position and suggests the best move using Stockfish.

---

## How it works

1. **Board detection** — Canny edges + contour search finds the board quad and warps it to a flat 800×800 image.
2. **Piece detection** — YOLOv8 (chess-specific weights) locates and classifies all 12 piece types.
3. **FEN construction** — Detections are mapped to squares and assembled into a FEN string (editable in the UI).
4. **Stockfish** — python-chess spawns Stockfish, analyses the FEN, and returns the best move + evaluation.

---

## Prerequisites

| Requirement | Install |
|---|---|
| Python 3.10–3.13 | python.org or homebrew |
| Stockfish | `brew install stockfish` (macOS) / `apt install stockfish` (Linux) |
| Chess-piece YOLO weights | `python download_model.py` (see below) |

---

## Setup

```bash
# 1. Clone / enter project directory
cd chess_solver

# 2. (Optional but recommended) create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Configure environment (optional — app auto-detects Stockfish on PATH)
cp .env.example .env
# Edit .env if Stockfish is not on PATH, or to tune engine settings

# 5. Download chess-piece YOLO weights
python download_model.py
```

> **Manual weights alternative:** If the download script fails, visit
> https://universe.roboflow.com and search "chess pieces". Export a YOLOv8 `.pt`
> model whose classes are `bB bK bN bP bQ bR wB wK wN wP wQ wR`.
> Save it to `models/chess_pieces_yolov8n.pt` (or set `YOLO_WEIGHTS` in `.env`).

---

## Run

```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

---

## UI controls

| Control | What it does |
|---|---|
| Board orientation | Rotate 180° if white pieces appear at top |
| Side to move | Tell the engine whose turn it is (can't be inferred from a photo) |
| Engine depth | Higher = stronger but slower (12 is good for demos) |
| Detection confidence | Lower = more detections, more false positives |
| Debug mode | Shows raw detections, FEN positions JSON, and engine PV line |

---

## Tips for good reads

- Shoot from directly above — avoid steep angles.
- Include all four edges of the board in the frame.
- Even lighting; avoid harsh shadows or glare.
- If pieces are misidentified, use the **FEN editor** in the UI to correct them.
- If the board quad isn't found, try cropping the photo closer to the board.

---

## Troubleshooting

**`Model does not appear to be a chess-piece model`**
→ Re-run `python download_model.py` or upload correct weights in the sidebar.

**`Stockfish not found`**
→ Install Stockfish, then set `STOCKFISH_PATH` in `.env` or the sidebar.

**Board not detected**
→ Better lighting, less angle, make sure all four corners are visible.

**Wrong pieces detected**
→ Adjust the confidence slider; edit the FEN manually; try re-shooting.

---

## Project structure

```
chess_solver/
├── streamlit_app.py      # Streamlit UI entry point
├── cv_utils.py           # Board detection, YOLO inference, FEN assembly
├── fen_utils.py          # FEN validation helpers
├── engine.py             # Stockfish subprocess wrapper
├── config.py             # Config / .env reader
├── download_model.py     # One-shot model downloader
├── tests/
│   ├── test_engine.py
│   ├── test_fen_utils.py
│   └── test_cv_utils.py
├── models/
│   └── chess_pieces_yolov8n.pt   ← you add this
├── assets/
│   └── sample_board.jpg
├── requirements.txt
├── .env.example
└── .env                  ← you create this (git-ignored)
```

---

## Known limitations

- **Side to move and castling rights** cannot be determined from the image alone — set them manually in the UI / FEN editor.
- Board detection fails on very angled or dark photos.
- Piece detection accuracy depends on the quality of the YOLO weights; a model trained on the specific piece style you're using will perform better.
- No GPU acceleration configured by default; first inference is slow (~2–3 s on CPU).

## License

MIT (project scaffolding). YOLO weights may carry their own license — check before distributing.
