"""Chess Solver — Image → Best Move via Stockfish."""
from __future__ import annotations
import base64
import hashlib
import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import chess
import chess.svg

import config
from engine import analyze_best_move_once
from fen_utils import validate_fen, sanitize_fen, board_from_fen, board_to_pretty, side_to_move as fen_side
from cv_utils import find_board_and_warp, detect_pieces_yolo, detect_pieces_classical, map_detections_to_fen

logging.basicConfig(level=logging.WARNING)


@st.cache_resource
def _piece_imgs() -> dict[str, str]:
    """SVG data URIs for all 12 piece types — cached across reruns."""
    imgs: dict[str, str] = {}
    for sym, (pt, c) in {
        'P': (chess.PAWN,   chess.WHITE), 'N': (chess.KNIGHT, chess.WHITE),
        'B': (chess.BISHOP, chess.WHITE), 'R': (chess.ROOK,   chess.WHITE),
        'Q': (chess.QUEEN,  chess.WHITE), 'K': (chess.KING,   chess.WHITE),
        'p': (chess.PAWN,   chess.BLACK), 'n': (chess.KNIGHT, chess.BLACK),
        'b': (chess.BISHOP, chess.BLACK), 'r': (chess.ROOK,   chess.BLACK),
        'q': (chess.QUEEN,  chess.BLACK), 'k': (chess.KING,   chess.BLACK),
    }.items():
        svg = chess.svg.piece(chess.Piece(pt, c), size=60)
        imgs[sym] = 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()
    return imgs


_MODEL_URL = (
    "https://huggingface.co/NAKSTStudio/yolov8m-chess-piece-detection"
    "/resolve/main/best.pt"
)
_MIN_MODEL_BYTES = 100_000

# Prefer the bundled models/ dir; fall back to /tmp (writable on all platforms)
_CANDIDATE_PATHS = [
    config.ROOT / "models" / "chess_pieces_yolov8n.pt",
    Path("/tmp/chess_pieces_yolov8n.pt"),
]


def _existing_model() -> Path | None:
    for p in _CANDIDATE_PATHS:
        if p.exists() and p.stat().st_size >= _MIN_MODEL_BYTES:
            return p
    return None


@st.cache_resource(show_spinner=False)
def _ensure_model() -> str | None:
    """Download YOLO model if absent. Returns path on success, None on failure."""
    existing = _existing_model()
    if existing:
        return str(existing)
    # Try writable locations in order
    for dest in _CANDIDATE_PATHS:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(2):
                try:
                    with urllib.request.urlopen(_MODEL_URL, timeout=180) as resp:
                        data = resp.read()
                    if len(data) >= _MIN_MODEL_BYTES:
                        dest.write_bytes(data)
                        return str(dest)
                except Exception:
                    if attempt == 0:
                        continue
        except OSError:
            continue  # directory not writable, try next candidate
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
        default_weights = str(_existing_model() or _CANDIDATE_PATHS[0])
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
if _existing_model() is None:
    with st.spinner("Downloading piece-detection model (~50 MB, one-time)…"):
        _ensure_model()
else:
    _ensure_model()  # warm the cache

st.divider()

# ---------------------------------------------------------------------------
# Step 1 — Image input
# ---------------------------------------------------------------------------
SAMPLE_PATH = config.ROOT / "assets" / "sample_board.jpg"

_game_active_now = st.session_state.get("game_active", False)
img_bgr = None
with st.expander("📷 Board image", expanded=not _game_active_now):
    tab_upload, tab_camera, tab_sample = st.tabs(["📁 Upload", "📷 Camera", "🖼 Sample"])

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
            col_s1, col_s2 = st.columns([1, 3])
            with col_s1:
                st.image(str(SAMPLE_PATH), width=115)
            with col_s2:
                st.caption("Sample board — use this to try without your own photo.")
                _b1, _b2, _bpad = st.columns([2, 1, 2])
                with _b1:
                    if st.button("Use sample", type="primary", use_container_width=True):
                        st.session_state["use_sample"] = True
                with _b2:
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

# flip_board tracks whether the photo was taken from Black's side.
# It always follows the side-to-move toggle: if Black moves, assume black is at bottom.
side_to_move_fen = st.session_state.get("side_move", "w")
flip_board = (side_to_move_fen == "b")

# Cache detections keyed by image content + side-to-move + conf threshold.
_img_hash = hashlib.md5(warped.tobytes()).hexdigest()[:16]
_det_cache_key = f"{_img_hash}_{side_to_move_fen}_{conf_thresh}"

if st.session_state.get("_det_cache_key") == _det_cache_key:
    detections = st.session_state["_detections"]
    detection_method = st.session_state["_det_method"]
    yolo_err_msg = st.session_state.get("_yolo_err")
else:
    # YOLO runs on original orientation; classical CV needs the rotated image
    with st.spinner("Detecting pieces…"):
        resolved_weights = _ensure_model() or yolo_weights_path
        detection_method = "YOLO"
        yolo_err_msg = None
        try:
            detections = detect_pieces_yolo(warped, weights_path=resolved_weights, conf_thresh=conf_thresh)
        except Exception as yolo_err:
            detection_method = "Classical CV"
            yolo_err_msg = f"{yolo_err.__class__.__name__}: {yolo_err}"
            warped_cv = cv2.rotate(warped, cv2.ROTATE_180) if flip_board else warped
            try:
                detections = detect_pieces_classical(warped_cv)
            except Exception as e:
                st.error(f"Detection failed: {e}")
                if debug_mode:
                    st.exception(e)
                st.stop()
    st.session_state["_det_cache_key"] = _det_cache_key
    st.session_state["_detections"] = detections
    st.session_state["_det_method"] = detection_method
    st.session_state["_yolo_err"] = yolo_err_msg

# Draw detection overlay on display image (rotated for classical CV, original for YOLO)
display_warped = cv2.rotate(warped, cv2.ROTATE_180) if (flip_board and detection_method == "Classical CV") else warped
overlay = display_warped.copy()
for d in detections:
    x1, y1, x2, y2 = map(int, d.xyxy)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 80), 2)
    cv2.putText(
        overlay, d.cls_name,
        (x1 + 2, max(y1 - 4, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 80), 1, cv2.LINE_AA,
    )

# FEN construction — for YOLO with flipped board, rotate the FEN instead of the image
mapped = map_detections_to_fen(warped, detections, side_to_move="w", flip=flip_board and detection_method == "YOLO")
pred_fen = mapped["fen"]
pred_fen, san_warnings = sanitize_fen(pred_fen)

pred_fen_parts = pred_fen.split()
pred_fen_parts[1] = side_to_move_fen
pred_fen = " ".join(pred_fen_parts)
side_label = "White" if side_to_move_fen == "w" else "Black"

# Detection method banner
method_html = (
    '<span class="method-badge badge-yolo">YOLO</span>'
    if detection_method == "YOLO"
    else '<span class="method-badge badge-cv">Classical CV (YOLO unavailable — accuracy ~50%)</span>'
)
st.markdown(
    f"Detection: {method_html} &nbsp;·&nbsp; **{len(detections)} pieces** found",
    unsafe_allow_html=True,
)

with st.expander(f"🔍 Detection details", expanded=False):
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown(f"**Warped board** {method_html}", unsafe_allow_html=True)
        st.image(cv2.cvtColor(display_warped, cv2.COLOR_BGR2RGB), use_container_width=True)
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
# Helpers
# ---------------------------------------------------------------------------
sf_path = stockfish_path_input.strip() or None

def _eval_html(cp, mate) -> str:
    if mate is not None:
        winner = "White" if mate > 0 else "Black"
        return f'<span class="eval-pill eval-mate">{winner} mates in {abs(mate)}</span>'
    if cp is not None:
        sign = "+" if cp >= 0 else ""
        css = "eval-white" if cp >= 0 else "eval-black"
        who = "White" if cp >= 0 else "Black"
        return f'<span class="eval-pill {css}">{who} {sign}{cp/100:.2f}</span>'
    return ""

def _board_svg(board: chess.Board, arrow: chess.svg.Arrow | None = None, size: int = 380) -> str:
    return chess.svg.board(
        board,
        arrows=[arrow] if arrow else [],
        flipped=(board.turn == chess.BLACK),
        size=size,
        style=".square.light{fill:#f0d9b5}.square.dark{fill:#b58863}",
        lastmove=arrow and chess.Move(arrow.tail, arrow.head),
    )

def _run_stockfish(board: chess.Board):
    try:
        return analyze_best_move_once(board, stockfish_path=sf_path, depth=depth, movetime_ms=movetime_ms)
    except FileNotFoundError as e:
        st.error(str(e))
        return None, None, None, {}
    except Exception as e:
        st.error(f"Stockfish error: {e}")
        return None, None, None, {}

# ---------------------------------------------------------------------------
# Validate detected FEN
# ---------------------------------------------------------------------------
valid_fen, fen_err = validate_fen(pred_fen)
if not valid_fen:
    st.warning(
        f"Position looks incomplete — {fen_err}.  \n"
        "This usually means the photo is blurry, the board was partially cropped, "
        "or the pieces weren't detected. Try retaking the photo or edit the FEN below."
    )

# ---------------------------------------------------------------------------
# Auto-start game as soon as valid FEN is detected (or FEN changes)
# ---------------------------------------------------------------------------
st.divider()

if valid_fen and pred_fen != st.session_state.get("game_initial_fen"):
    st.session_state["game_active"] = True
    st.session_state["game_board"] = board_from_fen(pred_fen)
    st.session_state["game_initial_fen"] = pred_fen
    st.session_state["game_history"] = []
    st.session_state["game_sf_move"] = None
    st.session_state["game_sel_sq"] = None
    st.session_state["_click_n"] = 0

if st.session_state.get("game_active"):
    game_board: chess.Board = st.session_state["game_board"]
    history: list = st.session_state["game_history"]

    # --- top controls ---
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 3])
    with ctrl1:
        if st.button("↩ Undo", use_container_width=True) and history:
            game_board.pop()
            history.pop()
            st.session_state["game_sf_move"] = None
            st.session_state["game_sel_sq"] = None
            st.toast("Move undone")
            st.rerun()
    with ctrl2:
        if st.button("⟳ Reset", use_container_width=True):
            st.session_state["game_board"] = board_from_fen(st.session_state["game_initial_fen"])
            st.session_state["game_history"] = []
            st.session_state["game_sf_move"] = None
            st.toast("Reset to detected position")
            st.rerun()
    with ctrl3:
        if st.button("✕ Exit game", use_container_width=True):
            st.session_state["game_active"] = False
            st.session_state["game_board"] = None
            st.session_state["game_history"] = []
            st.session_state["game_sf_move"] = None
            st.rerun()

    turn_label = "White" if game_board.turn == chess.WHITE else "Black"

    # Stockfish: only when a move was just made (not during piece selection)
    sf_move = st.session_state.get("game_sf_move")
    if sf_move is None and not game_board.is_game_over() and not st.session_state.get("game_sel_sq"):
        with st.spinner(f"Stockfish analysing for {turn_label}…"):
            best_move, cp, mate, _info = _run_stockfish(game_board)
        st.session_state["game_sf_move"] = (best_move, cp, mate)
    elif sf_move is not None:
        best_move, cp, mate = sf_move
    else:
        best_move = cp = mate = None

    # Board in a fragment so selection click only reruns the board (fast),
    # while move click does a full app rerun for Stockfish.
    @st.fragment
    def _board_fragment() -> None:
        _gb   = st.session_state["game_board"]
        _hist = st.session_state.get("game_history", [])
        _sfdat = st.session_state.get("game_sf_move")
        _best  = _sfdat[0] if _sfdat else None

        _sel = st.session_state.get("game_sel_sq")
        _sel_idx: chess.Square | None = None
        _vd: set[chess.Square] = set()
        if _sel:
            try:
                _sel_idx = chess.parse_square(_sel)
                for _m in _gb.legal_moves:
                    if _m.from_square == _sel_idx:
                        _vd.add(_m.to_square)
            except Exception:
                _sel = None; _sel_idx = None

        flipped = (_gb.turn == chess.BLACK)
        lm_from = lm_to = sf_from = sf_to = None
        if _hist:
            _lm = chess.Move.from_uci(_hist[-1])
            lm_from, lm_to = _lm.from_square, _lm.to_square
        if _best and not _sel:
            sf_from, sf_to = _best.from_square, _best.to_square

        imgs = _piece_imgs()
        fig = go.Figure()

        for _rank in range(8):
            for _file in range(8):
                _sq = chess.square(_file, _rank)
                _il = (_rank + _file) % 2 == 1
                px = (7 - _file) if flipped else _file
                py = (7 - _rank) if flipped else _rank
                if _sel_idx is not None and _sq == _sel_idx:
                    bg = "#f6f624"
                elif _sq in _vd:
                    bg = "#7fc97f"
                elif _sq in (lm_from, lm_to):
                    bg = "#cdd26a"
                elif _sq in (sf_from, sf_to):
                    bg = "#f6b25a"
                else:
                    bg = "#f0d9b5" if _il else "#b58863"
                fig.add_shape(type="rect", x0=px, x1=px+1, y0=py, y1=py+1,
                              fillcolor=bg, line_width=0, layer="below")
                _p = _gb.piece_at(_sq)
                if _p:
                    fig.add_layout_image(source=imgs[_p.symbol()],
                                         x=px+0.05, y=py+0.95, xref="x", yref="y",
                                         sizex=0.9, sizey=0.9,
                                         layer="above", sizing="stretch")

        for _dest in _vd:
            dpx = (7 - chess.square_file(_dest)) if flipped else chess.square_file(_dest)
            dpy = (7 - chess.square_rank(_dest)) if flipped else chess.square_rank(_dest)
            if _gb.piece_at(_dest):
                fig.add_shape(type="circle",
                              x0=dpx+0.05, x1=dpx+0.95, y0=dpy+0.05, y1=dpy+0.95,
                              line=dict(color="rgba(0,0,0,0.35)", width=5),
                              fillcolor="rgba(0,0,0,0)", layer="above")
            else:
                fig.add_shape(type="circle",
                              x0=dpx+0.32, x1=dpx+0.68, y0=dpy+0.32, y1=dpy+0.68,
                              fillcolor="rgba(0,0,0,0.28)", line_width=0, layer="above")

        for _i in range(8):
            fig.add_annotation(x=-0.3, y=_i+0.5, showarrow=False,
                               text=str(_i+1) if not flipped else str(8-_i),
                               font=dict(size=11, color="#7a6045"), xanchor="center")
            fig.add_annotation(x=_i+0.5, y=-0.3, showarrow=False,
                               text=chr(ord('a') + ((7-_i) if flipped else _i)),
                               font=dict(size=11, color="#7a6045"), yanchor="middle")

        sx, sy, sd = [], [], []
        for _rank in range(8):
            for _file in range(8):
                _sq = chess.square(_file, _rank)
                px = (7 - _file) if flipped else _file
                py = (7 - _rank) if flipped else _rank
                sx.append(px+0.5); sy.append(py+0.5)
                sd.append(chess.square_name(_sq))
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
                                 marker=dict(size=62, opacity=0.001,
                                             symbol="square", color="white"),
                                 customdata=sd,
                                 hovertemplate="<b>%{customdata}</b><extra></extra>",
                                 showlegend=False))
        fig.update_layout(
            width=560, height=560, margin=dict(l=25, r=5, t=5, b=25),
            xaxis=dict(range=[-0.6, 8.1], showgrid=False, zeroline=False,
                       showticklabels=False, fixedrange=True),
            yaxis=dict(range=[-0.6, 8.1], showgrid=False, zeroline=False,
                       showticklabels=False, fixedrange=True, scaleanchor="x"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            clickmode="event+select", dragmode=False, showlegend=False,
        )

        _cn = st.session_state.get("_click_n", 0)
        ev = st.plotly_chart(fig, on_select="rerun", key=f"cb_{_cn}",
                             use_container_width=True,
                             config={"displayModeBar": False, "scrollZoom": False})

        if (ev and getattr(ev, "selection", None)
                and ev.selection.points and not _gb.is_game_over()):
            clicked = ev.selection.points[0].get("customdata", "")
            if clicked:
                cur = st.session_state.get("game_sel_sq")
                if cur:
                    try:
                        fi = chess.parse_square(cur)
                        ti = chess.parse_square(clicked)
                        cands = sorted(
                            [m for m in _gb.legal_moves
                             if m.from_square == fi and m.to_square == ti],
                            key=lambda m: m.promotion or 0, reverse=True)
                        if cands:
                            _gb.push(cands[0])
                            _hist.append(cands[0].uci())
                            st.session_state["game_sf_move"] = None
                    except Exception:
                        pass
                    st.session_state["game_sel_sq"] = None
                    st.session_state["_click_n"] = _cn + 1
                    st.rerun(scope="app")   # full rerun → Stockfish recalculates
                else:
                    try:
                        ti = chess.parse_square(clicked)
                        tp = _gb.piece_at(ti)
                        if (tp and tp.color == _gb.turn and
                                any(m.from_square == ti for m in _gb.legal_moves)):
                            st.session_state["game_sel_sq"] = clicked
                        else:
                            st.session_state["game_sel_sq"] = None
                    except Exception:
                        st.session_state["game_sel_sq"] = None
                    st.session_state["_click_n"] = _cn + 1
                    st.rerun()              # fragment-only rerun → instant highlights

    # --- two-column layout ---
    col_brd, col_ctrl = st.columns([5, 2])
    with col_brd:
        _board_fragment()

    with col_ctrl:
        # Side-to-move toggle — next to the board
        _bw1, _bw2 = st.columns(2)
        with _bw1:
            if st.button("♙ White", use_container_width=True,
                         type="primary" if side_to_move_fen == "w" else "secondary"):
                st.session_state["side_move"] = "w"
                st.session_state["game_initial_fen"] = ""  # force re-init on new side
                st.rerun()
        with _bw2:
            if st.button("♟ Black", use_container_width=True,
                         type="primary" if side_to_move_fen == "b" else "secondary"):
                st.session_state["side_move"] = "b"
                st.session_state["game_initial_fen"] = ""  # force re-init on new side
                st.rerun()
        st.markdown("")

        if game_board.is_game_over():
            result = game_board.result()
            outcome = game_board.outcome()
            reason = outcome.termination.name.replace("_", " ").title() if outcome else ""
            st.success(f"Game over — **{result}** ({reason})")
        else:
            # Stockfish suggestion card
            if best_move:
                san = game_board.san(best_move)
                uci = best_move.uci()
                st.markdown(f"""
                <div class="move-card" style="padding:1.2rem 1.5rem;margin-bottom:1rem">
                    <div class="move-label">Stockfish suggests ({turn_label})</div>
                    <div class="move-san" style="font-size:2.2rem">{san}</div>
                    <div class="move-uci">{uci}</div>
                    {_eval_html(cp, mate)}
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"▶ Play {san}", type="primary", use_container_width=True):
                    game_board.push(best_move)
                    history.append(best_move.uci())
                    st.session_state["game_sel_sq"] = None
                    st.session_state["game_sf_move"] = None
                    st.rerun()

            # Move history
            if history:
                st.markdown("**Move history**")
                pairs = []
                for i in range(0, len(history), 2):
                    w = history[i]
                    b = history[i + 1] if i + 1 < len(history) else ""
                    pairs.append(f"{i//2 + 1}. {w}  {b}")
                st.code("\n".join(pairs), language="text")

else:
    # Not in game mode — show one-shot best move
    st.markdown("**Best move**")

    if valid_fen:
        board = board_from_fen(pred_fen)
        with st.spinner("Running Stockfish…"):
            best_move, cp, mate, _info = _run_stockfish(board)

        if best_move is None:
            st.warning("No legal moves — this may be checkmate or stalemate.")
        else:
            san = board.san(best_move)
            uci = best_move.uci()
            col_card, col_board = st.columns([1, 1])
            with col_card:
                st.markdown(f"""
                <div class="move-card">
                    <div class="move-label">Best move for {side_label}</div>
                    <div class="move-san">{san}</div>
                    <div class="move-uci">{uci}</div>
                    {_eval_html(cp, mate)}
                </div>
                """, unsafe_allow_html=True)
                if debug_mode and _info:
                    pv = [m.uci() for m in _info.get("pv", [])[:6]]
                    st.caption(f"PV: {' '.join(pv)}")
            with col_board:
                arrow = chess.svg.Arrow(best_move.from_square, best_move.to_square, color="#f97316")
                st.markdown(
                    f'<div style="display:flex;justify-content:center">{_board_svg(board, arrow)}</div>',
                    unsafe_allow_html=True,
                )

if debug_mode:
    with st.expander("Debug — raw detections"):
        st.json({sq: info[1] for sq, info in mapped["positions"].items()})

st.caption("Tip: if the move looks wrong, check the board orientation in ⚙️ Settings and verify the FEN above.")
