"""Board detection, piece detection, and FEN construction."""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

import config

logger = logging.getLogger(__name__)

LABEL_TO_FEN: Dict[str, str] = {
    "wK": "K", "wQ": "Q", "wR": "R", "wB": "B", "wN": "N", "wP": "P",
    "bK": "k", "bQ": "q", "bR": "r", "bB": "b", "bN": "n", "bP": "p",
}

_model_cache: Dict[str, "YOLO"] = {}


@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    conf: float
    cls_name: str


# ---------------------------------------------------------------------------
# Board detection
# ---------------------------------------------------------------------------

def _order_corners(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_board_and_warp(img_bgr: np.ndarray, out_size: int = 800) -> Optional[np.ndarray]:
    """Find the largest ~square contour and warp to top-down view."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(gray, 30, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        pts = approx.reshape(4, 2).astype(np.float32)
        rect = _order_corners(pts)
        tl, tr, br, bl = rect
        w_avg = (np.linalg.norm(br - bl) + np.linalg.norm(tr - tl)) / 2.0
        h_avg = (np.linalg.norm(tr - br) + np.linalg.norm(tl - bl)) / 2.0
        if h_avg == 0 or not (0.75 <= w_avg / h_avg <= 1.33):
            continue
        if cv2.contourArea(c) < 0.1 * img_area:
            continue
        dst = np.array(
            [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img_bgr, M, (out_size, out_size))

    return None


def split_into_squares(board_img: np.ndarray) -> List[np.ndarray]:
    h, w = board_img.shape[:2]
    sq_h, sq_w = h // 8, w // 8
    return [
        board_img[r * sq_h:(r + 1) * sq_h, c * sq_w:(c + 1) * sq_w]
        for r in range(8)
        for c in range(8)
    ]


# ---------------------------------------------------------------------------
# YOLO detection (primary)
# ---------------------------------------------------------------------------

def _validate_chess_model(model: "YOLO") -> bool:
    names = set(model.names.values()) if hasattr(model, "names") else set()
    return bool(names & config.CHESS_PIECE_LABELS)


_MIN_MODEL_BYTES = 100_000  # any valid .pt is at least 100 KB


def load_yolo_model(weights_path: str) -> "YOLO":
    if not _HAS_YOLO:
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
    if not weights_path or not os.path.isfile(weights_path):
        raise FileNotFoundError(f"YOLO weights not found: '{weights_path}'")
    size = os.path.getsize(weights_path)
    if size < _MIN_MODEL_BYTES:
        raise FileNotFoundError(
            f"Model file at '{weights_path}' is only {size} bytes — likely a failed download. "
            "Delete it and re-run python download_model.py."
        )
    if weights_path not in _model_cache:
        model = YOLO(weights_path)
        if not _validate_chess_model(model):
            names_sample = list(model.names.values())[:6]
            raise ValueError(
                f"Model is not a chess-piece model.\n"
                f"Got classes: {names_sample}…\n"
                f"Expected: {sorted(config.CHESS_PIECE_LABELS)}"
            )
        _model_cache[weights_path] = model
    return _model_cache[weights_path]


def detect_pieces_yolo(
    board_img: np.ndarray,
    weights_path: str,
    conf_thresh: float = 0.25,
) -> List[Detection]:
    model = load_yolo_model(weights_path)
    results = model.predict(source=board_img, conf=conf_thresh, verbose=False, imgsz=800)
    detections: List[Detection] = []
    for r in results:
        for b in r.boxes:
            xyxy = tuple(b.xyxy[0].cpu().numpy().tolist())
            conf = float(b.conf[0].cpu().numpy())
            cls_id = int(b.cls[0].cpu().numpy())
            name = r.names.get(cls_id, str(cls_id)) if hasattr(r, "names") else str(cls_id)
            if name.lower() == "board":
                continue
            detections.append(Detection(xyxy, conf, name))
    return detections


# ---------------------------------------------------------------------------
# Classical CV detection (fallback — no ML model needed)
# ---------------------------------------------------------------------------

def _classify_type(gray_patch: np.ndarray, piece_mask: np.ndarray) -> Tuple[str, float]:
    """
    Rough piece-type from silhouette mask.
    Works on the inner 60% of the square to avoid edge bleed.
    Expect 50-65 % accuracy on clean digital boards.
    """
    h, w = piece_mask.shape
    # Crop to inner region to ignore edge effects
    m = h // 5
    inner = piece_mask[m:-m, m:-m]
    ys, xs = np.where(inner > 0)
    if len(ys) == 0:
        return "P", 0.30

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    ph = max(y_max - y_min + 1, 1)
    pw = max(x_max - x_min + 1, 1)
    fill_ratio = inner.mean()
    aspect = ph / pw

    slices = np.array(
        [inner[y, x_min:x_max + 1].sum() for y in range(y_min, y_max + 1)],
        dtype=float,
    )
    slices /= slices.max() + 1e-6
    n = len(slices)
    top_w = slices[: max(n // 4, 1)].mean()
    mid_w = slices[n // 4: 3 * n // 4].mean()
    bot_w = slices[-max(n // 4, 1):].mean()

    # Small fill → pawn
    if fill_ratio < 0.06:
        return "P", 0.52
    # Tall narrow → bishop
    if aspect > 1.6 and top_w < 0.40:
        return "B", 0.48
    # Tall with wide top → queen
    if aspect > 1.4 and top_w > 0.45:
        return "Q", 0.45
    # Wide flat top, narrower body → rook (battlements)
    if top_w > 0.70 and top_w > bot_w * 0.9:
        return "R", 0.48
    # Large irregular mass → king or queen
    if fill_ratio > 0.22 and top_w > 0.45:
        return "K" if top_w > 0.55 else "Q", 0.42
    # Asymmetric top → knight
    if top_w < 0.35 and mid_w > top_w * 1.5:
        return "N", 0.40
    # Medium fill, medium aspect → pawn
    return "P", 0.40


def detect_pieces_classical(
    board_img: np.ndarray,
    std_thresh: float = 18.0,
) -> List[Detection]:
    """
    Classical CV piece detector — no ML model required.

    Uses a 33 % inner margin per square to avoid piece-icon bleed from
    neighbouring squares (digital boards render icons larger than one square).

    Presence: std-dev of the inner 34×34 crop (at 800px board).
      Empty flat squares → std<10; pieces add texture → std>18.
      Weak-signal squares (std<40) with a completely flat center are also
      skipped — they indicate neighbour-bleed rather than a real piece.
    Color: square-color-aware thresholds auto-tuned per board.
      Light-square bright threshold raised to 230 (vs 215) to avoid treating
      the board's near-white background as a white piece.
      n_br=0 AND n_dk=0 → ambiguous → skipped.
    Type: inner-region silhouette heuristics — ~50 % accuracy.
      Always use the FEN editor to verify and correct piece types.
    """
    gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
    h = board_img.shape[0]
    sq = h // 8
    m = sq // 3   # 33 % margin — wide enough to exclude icon bleed
    m4 = sq * 2 // 5  # 40 % margin — tighter crop for center-flatness check

    # Auto-detect which checkerboard parity is lighter (= light squares)
    avg = [0.0, 0.0]
    for row in range(8):
        for col in range(8):
            y0, y1 = row * sq + m, (row + 1) * sq - m
            x0, x1 = col * sq + m, (col + 1) * sq - m
            avg[(row + col) % 2] += float(gray[y0:y1, x0:x1].mean())
    light_parity = 0 if avg[0] >= avg[1] else 1

    detections: List[Detection] = []
    for row in range(8):
        for col in range(8):
            y0, y1 = row * sq, (row + 1) * sq
            x0, x1 = col * sq, (col + 1) * sq

            patch_gray = gray[y0:y1, x0:x1]
            inner = patch_gray[m:-m, m:-m]
            std3 = float(inner.std())

            if std3 < std_thresh:
                continue  # empty / flat square

            # Weak-signal square with a completely flat center → neighbour bleed
            center = patch_gray[m4:-m4, m4:-m4]
            ctr_std = float(center.std()) if center.size > 0 else 0.0
            if std3 < 40.0 and ctr_std < 5.0:
                continue

            is_light_sq = (row + col) % 2 == light_parity
            if is_light_sq:
                # Light bg ~215-225; raise bright threshold to 230 to avoid
                # treating bright board background as a white piece.
                n_bright = int((inner > 230).sum())
                n_dark = int((inner < 80).sum())
            else:
                # Dark bg ~125-140; white piece >155, black piece <55
                n_bright = int((inner > 155).sum())
                n_dark = int((inner < 55).sum())

            if n_bright == 0 and n_dark == 0:
                continue  # can't determine color — skip (likely bleed artifact)

            is_white = n_bright >= n_dark

            if is_white:
                piece_mask = (patch_gray > 190).astype(np.uint8)
            else:
                piece_mask = (patch_gray < 75).astype(np.uint8)

            color = "w" if is_white else "b"
            piece_type, conf = _classify_type(patch_gray, piece_mask)
            label = f"{color}{piece_type}"

            pad = sq // 10
            detections.append(Detection(
                xyxy=(x0 + pad, y0 + pad, x1 - pad, y1 - pad),
                conf=conf,
                cls_name=label,
            ))

    logger.debug("Classical detector found %d pieces.", len(detections))
    return detections


# ---------------------------------------------------------------------------
# FEN construction
# ---------------------------------------------------------------------------

def _detect_grid_boundaries(gray_board: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect actual column and row boundaries from the board image.

    Returns (col_bounds, row_bounds) — each an array of 9 pixel positions
    [0, b1, b2, ..., b7, W] (W = image width/height).

    Uses median of absolute pixel differences along each axis. The median is
    robust to piece-edge gradients (which only appear in a few rows/cols) and
    reliably picks up the alternating light/dark square transitions.
    """
    h, w = gray_board.shape
    gray_f = gray_board.astype(float)

    def _bounds_along(length: int, signal_1d: np.ndarray) -> np.ndarray:
        sq = length // 8
        smooth = np.convolve(signal_1d, np.ones(10) / 10, mode="same")
        bounds = [0]
        for i in range(1, 8):
            expected = i * sq
            lo = max(0, expected - 30)
            hi = min(length - 1, expected + 30)
            bounds.append(int(np.argmax(smooth[lo:hi])) + lo)
        bounds.append(length)
        return np.array(bounds, dtype=float)

    # Median across rows → robust column-boundary signal
    horiz_grad = np.abs(np.diff(gray_f, axis=1))
    col_signal = np.median(horiz_grad, axis=0)
    col_bounds = _bounds_along(w, col_signal)

    # Median across columns → robust row-boundary signal
    vert_grad = np.abs(np.diff(gray_f, axis=0))
    row_signal = np.median(vert_grad, axis=1)
    row_bounds = _bounds_along(h, row_signal)

    return col_bounds, row_bounds


def _normalize_label(raw: str) -> Optional[str]:
    if raw in LABEL_TO_FEN:
        return LABEL_TO_FEN[raw]
    for k, v in LABEL_TO_FEN.items():
        if k.lower() == raw.lower():
            return v
    raw_lower = raw.lower().replace("-", "").replace("_", "").replace(" ", "")
    color_map = {"white": "w", "black": "b", "w": "w", "b": "b"}
    piece_map = {
        "king": "K", "queen": "Q", "rook": "R", "bishop": "B", "knight": "N", "pawn": "P",
        "k": "K", "q": "Q", "r": "R", "b": "B", "n": "N", "p": "P",
    }
    for ck, cv in color_map.items():
        if raw_lower.startswith(ck):
            rest = raw_lower[len(ck):]
            piece = piece_map.get(rest)
            if piece:
                return LABEL_TO_FEN.get(cv + piece)
    logger.warning("Unrecognised piece label: %r", raw)
    return None


def map_detections_to_fen(
    board_img: np.ndarray,
    detections: List[Detection],
    side_to_move: str = "w",
    flip: bool = False,
) -> Dict:
    h, w = board_img.shape[:2]

    # Calibrate grid to actual square boundaries (handles boards with decorative frames)
    gray_board = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY) if board_img.ndim == 3 else board_img
    col_bounds, row_bounds = _detect_grid_boundaries(gray_board)

    positions: Dict[str, Tuple[float, str]] = {}
    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        # Use 60th-percentile x to compensate for leftward-biased bounding boxes
        # (chess piece icons extend further left than right in the NAKSTStudio model)
        cx = x1 + 0.60 * (x2 - x1)
        cy = (y1 + y2) / 2.0
        col = min(int(np.searchsorted(col_bounds[1:-1], cx)), 7)
        row = min(int(np.searchsorted(row_bounds[1:-1], cy)), 7)
        # If board is from black's perspective, mirror col and row to get standard coords
        if flip:
            col = 7 - col
            row = 7 - row
        sq = f"{chr(ord('a') + col)}{8 - row}"
        if sq not in positions or d.conf > positions[sq][0]:
            positions[sq] = (d.conf, d.cls_name)

    fen_rows = []
    for rank in range(8, 0, -1):
        row_str, empty = "", 0
        for col in range(8):
            sq = f"{chr(ord('a') + col)}{rank}"
            if sq in positions:
                if empty:
                    row_str += str(empty)
                    empty = 0
                piece = _normalize_label(positions[sq][1]) or "P"
                row_str += piece
            else:
                empty += 1
        if empty:
            row_str += str(empty)
        fen_rows.append(row_str)

    fen = f"{'/'.join(fen_rows)} {side_to_move} - - 0 1"
    return {"fen": fen, "positions": positions}
