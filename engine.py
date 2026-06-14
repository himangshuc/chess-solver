"""Stockfish service — one-shot: spawn, analyse, kill. Safe under Streamlit reruns."""
from __future__ import annotations
import logging
import os

import chess
import chess.engine

import config

logger = logging.getLogger(__name__)


def _resolve_path(override: str | None) -> str:
    """Return a valid Stockfish path or raise a clear FileNotFoundError."""
    if override and override.strip():
        p = override.strip()
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Stockfish binary not found at '{p}'. "
                "Check the path in the sidebar or update STOCKFISH_PATH in .env."
            )
        return p
    return config.stockfish_path()  # raises FileNotFoundError with helpful message if missing


def analyze_best_move_once(
    board: chess.Board,
    stockfish_path: str | None = None,
    depth: int = 12,
    movetime_ms: int | None = None,
) -> tuple[chess.Move | None, int | None, int | None, dict]:
    """
    Spawn Stockfish, analyse the position, return (best_move, cp, mate, info).
    Always closes the engine process — no zombie processes.
    """
    path = _resolve_path(stockfish_path)
    threads = config.engine_threads()
    hmb = config.engine_hash_mb()
    limit = (
        chess.engine.Limit(depth=depth)
        if not movetime_ms
        else chess.engine.Limit(time=movetime_ms / 1000.0)
    )

    logger.debug("Stockfish: %s  depth=%s  movetime_ms=%s", path, depth, movetime_ms)

    def _run(lim: chess.engine.Limit) -> tuple[chess.Move | None, int | None, int | None, dict]:
        with chess.engine.SimpleEngine.popen_uci(path) as eng:
            try:
                eng.configure({"Threads": threads, "Hash": hmb, "Ponder": False})
            except chess.engine.EngineError:
                pass  # some builds ignore certain options
            info = eng.analyse(board, lim)
            best = info.get("pv", [None])[0] or eng.play(board, lim).move
            cp, mate = _extract_eval(info, board)
            return best, cp, mate, info

    try:
        return _run(limit)
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError, BrokenPipeError):
        logger.warning("Stockfish crashed on first attempt; retrying with depth=8")
        try:
            return _run(chess.engine.Limit(depth=8))
        except Exception as e:
            logger.error("Stockfish failed on retry: %s", e)
            raise


def _extract_eval(info: dict, board: chess.Board) -> tuple[int | None, int | None]:
    score = info.get("score")
    if not score:
        return None, None
    # Always white's perspective: positive = white winning, negative = black winning
    white_pov = score.white()
    if white_pov.is_mate():
        return None, white_pov.mate()
    return white_pov.score(mate_score=100_000), None
