"""Interactive chess board Streamlit component backed by chessboard.js + chess.js."""
from __future__ import annotations
from pathlib import Path
import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent

_chess_board_fn = components.declare_component(
    "chess_board",
    path=str(_COMPONENT_DIR),
)


def chess_board(
    fen: str,
    *,
    sf_from: str | None = None,
    sf_to: str | None = None,
    last_from: str | None = None,
    last_to: str | None = None,
    flipped: bool = False,
    size: int = 400,
    key: str | None = None,
) -> str | None:
    """Render an interactive chess board.

    Returns the UCI move string (e.g. ``'e2e4'`` or ``'e7e8q'``) when the
    user makes a move, otherwise ``None``.

    Args:
        fen:       Current position as a FEN string.
        sf_from:   Square name of Stockfish suggestion source (orange).
        sf_to:     Square name of Stockfish suggestion target (orange).
        last_from: Square name of last move source (green).
        last_to:   Square name of last move target (green).
        size:      Board pixel size (width = height).
        key:       Streamlit component key.
    """
    return _chess_board_fn(
        fen=fen,
        sf_from=sf_from or "",
        sf_to=sf_to or "",
        last_from=last_from or "",
        last_to=last_to or "",
        flipped=flipped,
        size=size,
        key=key,
        default=None,
    )
