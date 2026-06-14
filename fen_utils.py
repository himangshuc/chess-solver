import chess


def validate_fen(fen: str) -> tuple[bool, str]:
    try:
        board = chess.Board(fen)
    except Exception as e:
        return False, str(e)

    piece_map = board.piece_map()
    whites = [p for p in piece_map.values() if p.color == chess.WHITE]
    blacks = [p for p in piece_map.values() if p.color == chess.BLACK]
    w_kings = sum(1 for p in whites if p.piece_type == chess.KING)
    b_kings = sum(1 for p in blacks if p.piece_type == chess.KING)

    if w_kings != 1:
        return False, f"Board must have exactly 1 white king (found {w_kings})"
    if b_kings != 1:
        return False, f"Board must have exactly 1 black king (found {b_kings})"
    if len(whites) > 16:
        return False, f"Too many white pieces: {len(whites)} (max 16)"
    if len(blacks) > 16:
        return False, f"Too many black pieces: {len(blacks)} (max 16)"

    return True, ""


def sanitize_fen(fen: str) -> tuple[str, list[str]]:
    """
    Best-effort fix of common FEN errors from classical CV detection.
    Returns (fixed_fen, list_of_warnings).
    Fixes: wrong king counts. Does NOT fix wrong piece types.
    """
    warnings: list[str] = []
    try:
        board = chess.Board(fen)
    except Exception as e:
        return fen, [f"FEN parse error: {e}"]

    piece_map = dict(board.piece_map())

    def _fix_kings(color: chess.Color) -> None:
        king_label = "white" if color == chess.WHITE else "black"
        piece_char = chess.KING
        king_squares = [sq for sq, p in piece_map.items()
                        if p.color == color and p.piece_type == piece_char]

        if len(king_squares) == 0:
            # Place king avoiding squares adjacent to the other king
            other_color = chess.BLACK if color == chess.WHITE else chess.WHITE
            other_kings = [sq for sq, p in piece_map.items()
                           if p.color == other_color and p.piece_type == chess.KING]
            other_king_sq = other_kings[0] if other_kings else None

            def _adjacent(a: int, b: int) -> bool:
                return chess.square_distance(a, b) <= 1

            default_sq = chess.E1 if color == chess.WHITE else chess.E8
            fallback = default_sq
            for sq in ([default_sq] + list(chess.SQUARES)):
                if sq not in piece_map and (other_king_sq is None or not _adjacent(sq, other_king_sq)):
                    fallback = sq
                    break
            piece_map[fallback] = chess.Piece(chess.KING, color)
            warnings.append(
                f"No {king_label} king detected — placed at {chess.square_name(fallback)}. "
                "Edit FEN to correct."
            )
        elif len(king_squares) > 1:
            # Keep first, demote rest to queen (most common misclassification)
            for sq in king_squares[1:]:
                piece_map[sq] = chess.Piece(chess.QUEEN, color)
            warnings.append(
                f"Multiple {king_label} kings detected — kept one at "
                f"{chess.square_name(king_squares[0])}, demoted rest to queens."
            )

    _fix_kings(chess.WHITE)
    _fix_kings(chess.BLACK)

    new_board = chess.Board(None)
    new_board.set_piece_map(piece_map)
    suffix = " ".join(fen.split()[1:]) if len(fen.split()) > 1 else "w - - 0 1"
    fixed = f"{new_board.board_fen()} {suffix}"
    return fixed, warnings


def board_from_fen(fen: str) -> chess.Board:
    return chess.Board(fen)


def board_to_pretty(board: chess.Board) -> str:
    return str(board)


def side_to_move(fen: str) -> str:
    try:
        parts = fen.split()
        return "White" if parts[1] == "w" else "Black"
    except Exception:
        return "Unknown"
