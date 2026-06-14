"""Tests for Stockfish service."""
import chess
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import analyze_best_move_once, _resolve_path


def test_stockfish_resolves():
    path = _resolve_path(None)
    assert os.path.isfile(path), f"Stockfish binary not found at {path}"


def test_stockfish_start_position():
    board = chess.Board()  # standard starting position
    best, cp, mate, info = analyze_best_move_once(board, depth=8)
    assert best is not None
    assert best in board.legal_moves, f"{best} is not a legal move"
    assert mate is None  # no mate from start


def test_stockfish_mate_in_one():
    # Ra8# — classic back-rank mate in 1
    # Black king b8, White king b6 controls escape squares, White rook plays Ra8#
    board = chess.Board("1k6/8/1K6/8/8/8/8/R7 w - - 0 1")
    best, cp, mate, _ = analyze_best_move_once(board, depth=10)
    assert best is not None
    assert mate is not None and mate > 0, "Expected mate-in-N score"


def test_bad_stockfish_path_raises():
    board = chess.Board()
    with pytest.raises(FileNotFoundError):
        analyze_best_move_once(board, stockfish_path="/nonexistent/stockfish", depth=6)


def test_stockfish_illegal_fen_raises():
    # python-chess will raise on an illegal position before we even call Stockfish
    with pytest.raises(Exception):
        chess.Board("invalid_fen")
