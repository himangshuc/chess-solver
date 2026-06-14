"""Interactive chess board Streamlit component."""
from __future__ import annotations
from pathlib import Path
import streamlit.components.v1 as components

_FRONTEND = Path(__file__).parent / "frontend"

_fn = components.declare_component("chess_board", path=str(_FRONTEND))


def chess_board(
    fen: str,
    *,
    sf_from: str | None = None,
    sf_to: str | None = None,
    last_from: str | None = None,
    last_to: str | None = None,
    flipped: bool = False,
    size: int = 420,
    key: str | None = None,
) -> str | None:
    """Render interactive chess board. Returns UCI move string on user move, else None."""
    return _fn(
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
