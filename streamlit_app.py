"""Chess Solver — Image → Best Move via Stockfish."""
from __future__ import annotations
import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import chess
import chess.svg

import config
from engine import analyze_best_move_once
from fen_utils import validate_fen, sanitize_fen, board_from_fen, board_to_pretty, side_to_move as fen_side
from cv_utils import find_board_and_warp, detect_pieces_yolo, detect_pieces_classical, map_detections_to_fen

logging.basicConfig(level=logging.WARNING)

_MODEL_URL = (
    "https://huggingface.co/NAKSTStudio/yolov8m-chess-piece-detection"
    "/resolve/main/best.pt"
)
_MODEL_PATH = config.ROOT / "models" / "chess_pieces_yolov8n.pt"
_MIN_MODEL_BYTES = 100_000


@st.cache_resource(show_spinner=False)
def _ensure_model() -> str | None:
    """Download YOLO model if absent. Returns path on success, None on failure."""
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _MODEL_PATH.exists() and _MODEL_PATH.stat().st_size >= _MIN_MODEL_BYTES:
        return str(_MODEL_PATH)
    try:
        with urllib.request.urlopen(_MODEL_URL, timeout=120) as resp:
            data = resp.read()
        if len(data) < _MIN_MODEL_BYTES:
            return None
        _MODEL_PATH.write_bytes(data)
        return str(_MODEL_PATH)
    except Exception:
        return None

st.set_page_config(
    page_title="Chess Solver",
    page_icon="♟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }
    .move-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid #0f3460;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .move-label { color: #a0aec0; font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem; }
    .move-san  { color: #ffffff; font-size: 3.5rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1; }
    .move-uci  { color: #718096; font-size: 1rem; margin-top: 0.4rem; font-family: monospace; }
    .eval-pill {
        display: inline-block;
        padding: 0.25rem 0.9rem;
        border-radius: 999px;
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.75rem;
    }
    .eval-white { background: #2d6a4f; color: #d8f3dc; }
    .eval-black { background: #7b2d00; color: #ffe0cc; }
    .eval-mate  { background: #2c3e50; color: #f39c12; }
    .step-badge {
        display: inline-flex; align-items: center; justify-content: center;
        width: 28px; height: 28px; border-radius: 50%;
        background: #0f3460; color: white; font-weight: 700;
        font-size: 0.8rem; margin-right: 0.5rem;
    }
    .method-badge {
        display: inline-block; padding: 0.2rem 0.7rem;
        border-radius: 6px; font-size: 0.75rem; font-weight: 600;
    }
    .badge-yolo { background: #1a4731; color: #68d391; }
    .badge-cv   { background: #4a2500; color: #fbd38d; }
    div[data-testid="stRadio"] > div { gap: 0.5rem; }
    div[data-testid="stRadio"] label {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 8px; padding: 0.5rem 1.2rem;
        cursor: pointer; transition: all 0.15s;
    }
    div[data-testid="stRadio"] label:hover { border-color: #60a5fa; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — advanced settings only
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Advanced Settings")

    with st.expander("Engine", expanded=True):
        try:
            default_sf = config.stockfish_path()
        except FileNotFoundError:
            default_sf = ""
        stockfish_path_input = st.text_input(
            "Stockfish path",
            value=default_sf,
            help="Leave blank to auto-detect from PATH / STOCKFISH_PATH env var.",
        )
        depth = st.slider("Search depth", 6, 20, 12)
        movetime_ms = st.slider("Max time per move (ms)", 500, 10_000, 3_000, step=250)

    with st.expander("Detection"):
        default_weights = str(_MODEL_PATH)
        yolo_weights_path = st.text_input("YOLO weights (.pt)", value=default_weights)
        uploaded_weights = st.file_uploader("Upload custom weights", type=["pt"])
        if uploaded_weights is not None:
            save_path = str(config.ROOT / "models" / uploaded_weights.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_weights.read())
            yolo_weights_path = save_path
            st.success(f"Saved → {save_path}")
        conf_thresh = st.slider("Confidence threshold", 0.10, 0.80, 0.25, step=0.05)

    debug_mode = st.checkbox("Debug mode", value=config.debug_mode())

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# ♟ Chess Solver")
st.markdown("**Upload a board photo — get the best move instantly.**")

# Auto-download model in background on first load
if not (_MODEL_PATH.exists() and _MODEL_PATH.stat().st_size >= _MIN_MODEL_BYTES):
    with st.spinner("Downloading piece-detection model (~50 MB, one-time)…"):
        _ensure_model()
else:
    _ensure_model()  # warm the cache

st.divider()

# ---------------------------------------------------------------------------
# Step 1 — Image input
# ---------------------------------------------------------------------------
st.markdown('<span class="step-badge">1</span> **Choose a board image**', unsafe_allow_html=True)

SAMPLE_PATH = config.ROOT / "assets" / "sample_board.jpg"
tab_upload, tab_camera, tab_sample = st.tabs(["📁 Upload", "📷 Camera", "🖼 Sample"])

img_bgr = None

with tab_upload:
    file = st.file_uploader(
        "Drop a chessboard photo here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if file is not None:
        buf = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        st.session_state.pop("use_sample", None)

with tab_camera:
    cam = st.camera_input("Take a snapshot", label_visibility="collapsed")
    if cam is not None:
        buf = np.frombuffer(cam.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        st.session_state.pop("use_sample", None)

with tab_sample:
    if SAMPLE_PATH.exists():
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            st.image(str(SAMPLE_PATH), use_container_width=True)
        with col_s2:
            st.markdown("**Sample board**\n\nUse this to try the solver without your own photo.")
            if st.button("Use sample", type="primary", use_container_width=True):
                st.session_state["use_sample"] = True
            if st.button("Clear", use_container_width=True):
                st.session_state.pop("use_sample", None)
    else:
        st.info("No sample image found. Add one to `assets/sample_board.jpg`")

if st.session_state.get("use_sample") and SAMPLE_PATH.exists():
    img_bgr = cv2.imread(str(SAMPLE_PATH))

if img_bgr is None:
    st.markdown("")
    st.info("⬆ Upload a photo, take one with your camera, or try the sample image.")
    st.stop()

# ---------------------------------------------------------------------------
# Board detection + piece detection (run together, show progress)
# ---------------------------------------------------------------------------
with st.spinner("Analyzing board…"):
    warped = find_board_and_warp(img_bgr, out_size=800)

if warped is None:
    st.error(
        "**Could not detect the board.**\n\n"
        "Make sure the full board is visible, lighting is even, and the photo isn't at a steep angle."
    )
    st.stop()

flip_board = st.session_state.get("flip_board", False)
if flip_board:
    warped = cv2.rotate(warped, cv2.ROTATE_180)

detection_method = "YOLO"
yolo_err_msg = None
with st.spinner("Detecting pieces…"):
    resolved_weights = _ensure_model() or yolo_weights_path
    try:
        detections = detect_pieces_yolo(warped, weights_path=resolved_weights, conf_thresh=conf_thresh)
    except Exception as yolo_err:
        detection_method = "Classical CV"
        yolo_err_msg = f"{yolo_err.__class__.__name__}: {yolo_err}"
        try:
            detections = detect_pieces_classical(warped)
        except Exception as e:
            st.error(f"Detection failed: {e}")
            if debug_mode:
                st.exception(e)
            st.stop()

# Draw detection overlay
overlay = warped.copy()
for d in detections:
    x1, y1, x2, y2 = map(int, d.xyxy)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 80), 2)
    cv2.putText(
        overlay, d.cls_name,
        (x1 + 2, max(y1 - 4, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 80), 1, cv2.LINE_AA,
    )

# FEN construction
mapped = map_detections_to_fen(warped, detections, side_to_move="w")
pred_fen = mapped["fen"]
if detection_method != "YOLO":
    pred_fen, san_warnings = sanitize_fen(pred_fen)
else:
    san_warnings = []

# ---------------------------------------------------------------------------
# Step 2 — Board orientation, side to move, color swap
# ---------------------------------------------------------------------------
st.divider()
st.markdown('<span class="step-badge">2</span> **Set up the view**', unsafe_allow_html=True)

col_orient_w, col_orient_b, col_side_w, col_side_b, col_swap = st.columns([1, 1, 1, 1, 2])

with col_orient_w:
    orient_w_btn = st.button(
        "⬜ White at bottom",
        use_container_width=True,
        type="primary" if not st.session_state.get("flip_board", False) else "secondary",
        help="Standard view — white pieces at the bottom.",
    )
with col_orient_b:
    orient_b_btn = st.button(
        "⬛ Black at bottom",
        use_container_width=True,
        type="primary" if st.session_state.get("flip_board", False) else "secondary",
        help="Flip if the photo was taken from Black's side.",
    )
with col_side_w:
    white_btn = st.button("♙ White moves", use_container_width=True,
                          type="primary" if st.session_state.get("side_move", "w") == "w" else "secondary")
with col_side_b:
    black_btn = st.button("♟ Black moves", use_container_width=True,
                          type="primary" if st.session_state.get("side_move", "w") == "b" else "secondary")
with col_swap:
    swap_btn = st.button(
        "⇄ Swap piece colors",
        use_container_width=True,
        help="Use if white/black piece labels look reversed.",
    )

if orient_w_btn:
    st.session_state["flip_board"] = False
if orient_b_btn:
    st.session_state["flip_board"] = True
if white_btn:
    st.session_state["side_move"] = "w"
if black_btn:
    st.session_state["side_move"] = "b"
if swap_btn:
    st.session_state["swap_colors"] = not st.session_state.get("swap_colors", False)

side_to_move_fen = st.session_state.get("side_move", "w")
side_label = "White" if side_to_move_fen == "w" else "Black"

# Swap piece colors in FEN if toggled (invert uppercase ↔ lowercase in piece placement)
if st.session_state.get("swap_colors", False):
    placement = pred_fen.split()[0]
    swapped = "".join(c.lower() if c.isupper() else c.upper() if c.islower() else c for c in placement)
    pred_fen_parts = pred_fen.split()
    pred_fen_parts[0] = swapped
    pred_fen = " ".join(pred_fen_parts)
    st.info("⇄ Piece colors swapped — white and black labels are exchanged.")

pred_fen_parts = pred_fen.split()
pred_fen_parts[1] = side_to_move_fen
pred_fen = " ".join(pred_fen_parts)

# Detection images + warnings in collapsed expander
method_html = (
    '<span class="method-badge badge-yolo">YOLO</span>'
    if detection_method == "YOLO"
    else '<span class="method-badge badge-cv">Classical CV</span>'
)
with st.expander(f"🔍 Detection details — {len(detections)} pieces found", expanded=False):
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown(f"**Warped board** {method_html}", unsafe_allow_html=True)
        st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_img2:
        st.markdown("**Piece overlay**")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
    if yolo_err_msg:
        st.warning(
            "**YOLO model unavailable** — piece *type* detection is approximate (~50%).\n\n"
            "Run `python download_model.py` to auto-download the public model, or grab `best.pt` from "
            "[NAKSTStudio/yolov8m-chess-piece-detection](https://huggingface.co/NAKSTStudio/yolov8m-chess-piece-detection) "
            f"and place it at `models/chess_pieces_yolov8n.pt`\n\nError: `{yolo_err_msg}`"
        )

for w in san_warnings:
    st.warning(f"Auto-fix: {w}")

# ---------------------------------------------------------------------------
# Step 3 — Stockfish analysis
# ---------------------------------------------------------------------------
st.divider()
st.markdown('<span class="step-badge">3</span> **Best move**', unsafe_allow_html=True)

sf_path = stockfish_path_input.strip() or None

valid_fen, fen_err = validate_fen(pred_fen)
if not valid_fen:
    st.error(f"Invalid position detected — {fen_err}. Edit the FEN below and retry.")
else:
    board = board_from_fen(pred_fen)
    with st.spinner("Running Stockfish…"):
        try:
            best_move, cp, mate, _info = analyze_best_move_once(
                board,
                stockfish_path=sf_path,
                depth=depth,
                movetime_ms=movetime_ms,
            )
        except FileNotFoundError as e:
            st.error(str(e))
            best_move = None
            cp = mate = _info = None
        except Exception as e:
            st.error(f"Stockfish error: {e}")
            st.caption("The position may be illegal. Edit the FEN below and retry.")
            best_move = None
            cp = mate = _info = None

    if best_move is None and valid_fen:
        st.warning("No legal moves — this may be checkmate or stalemate.")
    elif best_move is not None:
        san = board.san(best_move)
        uci = best_move.uci()

        # Eval pill — cp is always from white's perspective (standard convention)
        if mate is not None:
            winner = "White" if mate > 0 else "Black"
            eval_html = f'<span class="eval-pill eval-mate">{winner} mates in {abs(mate)}</span>'
        elif cp is not None:
            sign = "+" if cp >= 0 else ""
            # positive cp = white winning (green), negative = black winning (red)
            css = "eval-white" if cp >= 0 else "eval-black"
            who = "White" if cp >= 0 else "Black"
            eval_html = f'<span class="eval-pill {css}">{who} {sign}{cp/100:.2f}</span>'
        else:
            eval_html = ""

        col_card, col_board = st.columns([1, 1])

        with col_card:
            st.markdown(f"""
            <div class="move-card">
                <div class="move-label">Best move for {side_label}</div>
                <div class="move-san">{san}</div>
                <div class="move-uci">{uci}</div>
                {eval_html}
            </div>
            """, unsafe_allow_html=True)
            if debug_mode and _info:
                pv = [m.uci() for m in _info.get("pv", [])[:6]]
                st.caption(f"PV: {' '.join(pv)}")

        with col_board:
            arrow = chess.svg.Arrow(
                best_move.from_square,
                best_move.to_square,
                color="#f97316",
            )
            svg = chess.svg.board(
                board,
                arrows=[arrow],
                flipped=flip_board,
                size=380,
                style=(
                    ".square.light { fill: #f0d9b5; }"
                    ".square.dark  { fill: #b58863; }"
                ),
            )
            st.markdown(
                f'<div style="display:flex;justify-content:center">{svg}</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# FEN editor (collapsible)
# ---------------------------------------------------------------------------
st.divider()
with st.expander("✏️ Edit position (FEN)", expanded=not valid_fen):
    col_fen_l, col_fen_r = st.columns([3, 2])
    with col_fen_l:
        fen_input = st.text_input(
            "FEN string",
            value=pred_fen,
            help="Format: <placement> <w/b> <castling> <ep> <halfmove> <fullmove>",
        )
        v2, e2 = validate_fen(fen_input)
        if not v2:
            st.error(f"Invalid FEN — {e2}")
        else:
            st.success("Valid position")
            b2 = board_from_fen(fen_input)
            with st.expander("Board preview", expanded=False):
                st.code(board_to_pretty(b2), language="text")

    with col_fen_r:
        st.markdown("**FEN piece letters**")
        st.markdown(
            "| | White | Black |\n"
            "|--|--|--|\n"
            "| King | `K` | `k` |\n"
            "| Queen | `Q` | `q` |\n"
            "| Rook | `R` | `r` |\n"
            "| Bishop | `B` | `b` |\n"
            "| Knight | `N` | `n` |\n"
            "| Pawn | `P` | `p` |\n\n"
            "Digits = empty squares. Ranks separated by `/`."
        )

if debug_mode:
    with st.expander("Debug — raw detections"):
        st.json({sq: info[1] for sq, info in mapped["positions"].items()})

st.caption("Tip: if the move looks wrong, check the board orientation in ⚙️ Settings and verify the FEN above.")
