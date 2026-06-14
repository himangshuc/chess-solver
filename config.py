"""Central config — reads .env, provides typed accessors with sane defaults."""
from __future__ import annotations
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent


def stockfish_path() -> str:
    """Return the Stockfish binary path, checking common locations."""
    env = os.getenv("STOCKFISH_PATH", "").strip()
    if env and Path(env).is_file():
        return env
    # Auto-detect from PATH
    found = shutil.which("stockfish")
    if found:
        return found
    # apt installs to /usr/games on Debian/Ubuntu (Streamlit Cloud)
    for candidate in ["/usr/games/stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish"]:
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        "Stockfish not found. Install it (e.g. brew install stockfish) "
        "and set STOCKFISH_PATH in .env, or ensure it is on your PATH."
    )


def yolo_weights_path() -> Path:
    env = os.getenv("YOLO_WEIGHTS", "models/chess_pieces_yolov8n.pt").strip()
    return ROOT / env if not Path(env).is_absolute() else Path(env)


def engine_threads() -> int:
    return int(os.getenv("ENGINE_THREADS", "1"))


def engine_hash_mb() -> int:
    return int(os.getenv("ENGINE_HASH_MB", "32"))


def debug_mode() -> bool:
    return os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")


# Expected class names for a chess-piece YOLO model.
# Accepts both compact format (wK, bP) and long format (white_king, black_pawn).
CHESS_PIECE_LABELS = {
    "bB", "bK", "bN", "bP", "bQ", "bR", "wB", "wK", "wN", "wP", "wQ", "wR",
    "white_king", "white_queen", "white_rook", "white_bishop", "white_knight", "white_pawn",
    "black_king", "black_queen", "black_rook", "black_bishop", "black_knight", "black_pawn",
}
