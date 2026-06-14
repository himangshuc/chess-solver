"""
Download a chess-piece YOLOv8 model.

Primary model: NAKSTStudio/yolov8m-chess-piece-detection (public, no auth needed).

Three options (tried in order):
  A. HuggingFace — public model, no token required:
       https://huggingface.co/NAKSTStudio/yolov8m-chess-piece-detection

  B. Roboflow API — requires a free account + ROBOFLOW_API_KEY env var:
       https://universe.roboflow.com/roboflow-100/chess-pieces-mjzgj

  C. Manual — download from HuggingFace in your browser, then move
       the .pt file to  models/chess_pieces_yolov8n.pt

Usage:
    # Option A (default — no credentials needed):
    python download_model.py

    # Option B: set ROBOFLOW_API_KEY, then run:
    ROBOFLOW_API_KEY=xxxx python download_model.py

    # Custom output path:
    python download_model.py --out models/my_chess_model.pt
"""
from __future__ import annotations
import argparse
import os
import sys
import urllib.request
from pathlib import Path

DEFAULT_OUT = Path(__file__).parent / "models" / "chess_pieces_yolov8n.pt"
MIN_MODEL_BYTES = 100_000

# HuggingFace — public model, no token required
HF_URL = (
    "https://huggingface.co/NAKSTStudio/yolov8m-chess-piece-detection"
    "/resolve/main/best.pt"
)

MANUAL_INSTRUCTIONS = """
╔═══════════════════════════════════════════════════════════════════╗
║           MANUAL MODEL DOWNLOAD INSTRUCTIONS                       ║
╠═══════════════════════════════════════════════════════════════════╣
║  Automated download failed. Get the model manually (free):         ║
║                                                                    ║
║  OPTION A — HuggingFace direct download (easiest, no login):       ║
║    1. Visit:                                                        ║
║       https://huggingface.co/NAKSTStudio/yolov8m-chess-piece-detection
║    2. Click "Files and versions" → download best.pt (~50 MB)       ║
║    3. Move the .pt file to:                                        ║
║       {out}  ║
║                                                                    ║
║  OPTION B — Roboflow Universe (browser):                           ║
║    1. Visit https://universe.roboflow.com                          ║
║    2. Search "chess pieces mjzgj"                                  ║
║    3. Click "Model" → Export → YOLOv8 PyTorch → Download          ║
║    4. Move the .pt file to:                                        ║
║       {out}  ║
║                                                                    ║
║  OPTION C — Roboflow Python API:                                   ║
║    1. pip install roboflow                                         ║
║    2. Get a free API key from https://app.roboflow.com             ║
║    3. Re-run:  ROBOFLOW_API_KEY=your_key python download_model.py  ║
║                                                                    ║
║  Until you have a model, the app uses classical CV (presence +     ║
║  color are reliable; piece TYPE is approximate ~50% accuracy).     ║
║  Correct the FEN manually in the app's text editor before         ║
║  running Stockfish.                                                ║
╚═══════════════════════════════════════════════════════════════════╝
"""


def _try_huggingface(out: Path) -> bool:
    token = os.environ.get("HF_TOKEN", "").strip()
    print("Trying HuggingFace download (public model, no token required) …")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(HF_URL, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp, open(str(out), "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 / total
                    print(f"\r  {pct:.1f}% ({downloaded/1e6:.1f} MB)", end="", flush=True)
        print()
        size = out.stat().st_size
        if size < MIN_MODEL_BYTES:
            out.unlink(missing_ok=True)
            print(f"  ✗ HuggingFace download: file too small ({size} bytes) — invalid token?")
            return False
        print(f"  ✓ Saved → {out} ({size/1e6:.1f} MB)")
        return True
    except Exception as e:
        out.unlink(missing_ok=True)
        print(f"  ✗ HuggingFace failed: {e}")
        return False


def _try_roboflow(out: Path) -> bool:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        return False
    print("Trying Roboflow API download (ROBOFLOW_API_KEY found) …")
    try:
        from roboflow import Roboflow  # type: ignore
    except ImportError:
        print("  roboflow package not installed. Run: pip install roboflow")
        return False
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("roboflow-100").project("chess-pieces-mjzgj")
        version = project.version(1)
        # Export the model weights — downloads to ./chess-pieces-mjzgj-1/
        version.download("yolov8")
        # Find the exported best.pt
        for candidate in Path(".").rglob("best.pt"):
            size = candidate.stat().st_size
            if size >= MIN_MODEL_BYTES:
                candidate.rename(out)
                print(f"  ✓ Moved model → {out}")
                return True
        print("  ✗ Downloaded but could not locate best.pt")
        return False
    except Exception as e:
        print(f"  ✗ Roboflow API failed: {e}")
        return False


def verify(out: Path) -> None:
    try:
        from ultralytics import YOLO
        from config import CHESS_PIECE_LABELS
    except ImportError:
        print("ultralytics not installed — skipping model verification.")
        return
    m = YOLO(str(out))
    names = set(m.names.values())
    overlap = names & CHESS_PIECE_LABELS
    if not overlap:
        print(
            f"WARNING: model classes {sorted(names)[:6]}… don't match chess piece labels.\n"
            "The model may not work. Expected classes: "
            + str(sorted(CHESS_PIECE_LABELS))
        )
    else:
        print(f"Model verified ✓ — {len(names)} classes, chess labels found: {sorted(overlap)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download chess piece YOLOv8 model")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output .pt path")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and out.stat().st_size >= MIN_MODEL_BYTES:
        print(f"Model already exists at {out} ({out.stat().st_size/1e6:.1f} MB).")
        if not args.skip_verify:
            verify(out)
        sys.exit(0)

    out.parent.mkdir(parents=True, exist_ok=True)

    if _try_huggingface(out) or _try_roboflow(out):
        if not args.skip_verify:
            verify(out)
        sys.exit(0)

    # Neither token was found — print manual instructions
    pad = " " * max(0, 68 - len(str(out)))
    print(MANUAL_INSTRUCTIONS.format(out=str(out) + pad))
    sys.exit(1)
