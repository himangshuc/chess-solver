"""Tests for board detection, square splitting, FEN assembly, and label normalisation."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cv_utils import (
    find_board_and_warp,
    split_into_squares,
    map_detections_to_fen,
    Detection,
    _normalize_label,
    LABEL_TO_FEN,
)


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", list(LABEL_TO_FEN.items()))
def test_normalize_exact_labels(raw, expected):
    assert _normalize_label(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("white-queen", "Q"),
    ("black_rook", "r"),
    ("White_King", "K"),
    ("bk", "k"),   # lowercase two-char shorthand
    ("wP", "P"),
])
def test_normalize_variant_labels(raw, expected):
    result = _normalize_label(raw)
    assert result == expected, f"normalize_label({raw!r}) = {result!r}, want {expected!r}"


def test_normalize_unknown_label_returns_none():
    assert _normalize_label("banana") is None


# ---------------------------------------------------------------------------
# Square splitting
# ---------------------------------------------------------------------------

def test_split_into_squares_count():
    board = np.zeros((800, 800, 3), dtype=np.uint8)
    squares = split_into_squares(board)
    assert len(squares) == 64


def test_split_into_squares_shape():
    board = np.zeros((800, 800, 3), dtype=np.uint8)
    for sq in split_into_squares(board):
        assert sq.shape == (100, 100, 3)


# ---------------------------------------------------------------------------
# Board detection on synthetic image
# ---------------------------------------------------------------------------

def _make_board_image(size: int = 600) -> np.ndarray:
    """Create a white image with a black square drawn as a stand-in chessboard quad."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 200
    margin = size // 6
    pts = np.array([
        [margin, margin],
        [size - margin, margin],
        [size - margin, size - margin],
        [margin, size - margin],
    ], dtype=np.int32)
    cv2 = __import__("cv2")
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=4)
    return img


def test_find_board_warp_returns_square():
    img = _make_board_image()
    warped = find_board_and_warp(img, out_size=400)
    assert warped is not None
    assert warped.shape[0] == warped.shape[1] == 400


def test_find_board_warp_blank_returns_none():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 180  # featureless grey
    result = find_board_and_warp(img)
    assert result is None


# ---------------------------------------------------------------------------
# FEN assembly
# ---------------------------------------------------------------------------

def test_map_detections_empty():
    board_img = np.zeros((800, 800, 3), dtype=np.uint8)
    result = map_detections_to_fen(board_img, [], side_to_move="w")
    fen = result["fen"]
    # empty board = 8 rows of 8 empty squares
    assert fen.startswith("8/8/8/8/8/8/8/8")
    assert " w " in fen


def test_map_detections_single_piece():
    board_img = np.zeros((800, 800, 3), dtype=np.uint8)
    # Place a white king at e1 — center of col 4 (e), row 7 (rank 1), pixel ~(450, 750)
    sq_size = 800 / 8
    cx = 4 * sq_size + sq_size / 2  # e-file
    cy = 7 * sq_size + sq_size / 2  # rank 1 (bottom row)
    d = Detection(xyxy=(cx - 10, cy - 10, cx + 10, cy + 10), conf=0.9, cls_name="wK")
    result = map_detections_to_fen(board_img, [d], side_to_move="b")
    fen = result["fen"]
    # last rank (rank 1) should contain K
    last_rank = fen.split(" ")[0].split("/")[-1]
    assert "K" in last_rank
    assert " b " in fen


def test_map_detections_side_to_move():
    board_img = np.zeros((800, 800, 3), dtype=np.uint8)
    for stm, expected in [("w", " w "), ("b", " b ")]:
        fen = map_detections_to_fen(board_img, [], side_to_move=stm)["fen"]
        assert expected in fen
