"""Tests for FEN validation and helpers."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fen_utils import validate_fen, board_from_fen, side_to_move


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
SCHOLARS_MATE = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"


def test_valid_starting_fen():
    ok, err = validate_fen(STARTING_FEN)
    assert ok, err


def test_valid_scholars_mate():
    ok, err = validate_fen(SCHOLARS_MATE)
    assert ok, err


def test_invalid_fen_empty():
    ok, err = validate_fen("")
    assert not ok


def test_invalid_fen_garbage():
    ok, err = validate_fen("not_a_fen at all")
    assert not ok


def test_side_to_move_white():
    assert side_to_move(STARTING_FEN) == "White"


def test_side_to_move_black():
    assert side_to_move(SCHOLARS_MATE) == "Black"


def test_board_from_fen_legal_moves():
    import chess
    board = board_from_fen(STARTING_FEN)
    assert len(list(board.legal_moves)) == 20  # 20 legal moves from start
