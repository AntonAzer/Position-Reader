"""
Thin wrapper around the Stockfish UCI engine (via python-chess) that takes
a FEN and returns the best move, best line (principal variation) in SAN,
and an evaluation.
"""
import shutil
import chess
import chess.engine

STOCKFISH_CANDIDATES = [
    "./stockfish-windows-x86-64-avx2.exe",
    "/usr/games/stockfish",
    "/usr/bin/stockfish",
    "stockfish",
]


def find_stockfish_path():
    for candidate in STOCKFISH_CANDIDATES:
        resolved = shutil.which(candidate) or (candidate if candidate.startswith("/") else None)
        if resolved:
            import os
            if os.path.isfile(resolved) and os.access(resolved, os.X_OK):
                return resolved
    raise FileNotFoundError(
        "Stockfish binary not found. Install it (e.g. `apt-get install stockfish`) "
        "or set STOCKFISH_PATH."
    )


def analyze_fen(fen, depth=18, movetime_ms=None, multipv=1):
    """
    Returns a dict:
      {
        "fen": <input fen>,
        "turn": "white"/"black",
        "eval": {"type": "cp"|"mate", "value": <int>},   # from side-to-move's POV
        "best_move_uci": "...",
        "best_move_san": "...",
        "best_line_san": ["e4", "e5", "Nf3", ...],
        "lines": [ { same shape, for multipv } ]   # only if multipv > 1
      }
    Raises ValueError if the FEN is illegal.
    """
    try:
        board = chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"Invalid FEN: {e}")

    # Vision-detected (or hand-edited) castling rights are a guess -- if a
    # right doesn't actually match the rook/king placement, drop just that
    # right rather than rejecting the whole position.
    board.castling_rights = board.clean_castling_rights()

    if not board.is_valid():
        status = board.status()
        raise ValueError(
            "This position is not a legal chess position (e.g. wrong number "
            "of kings, a king that could be captured, or pawns on rank 1/8). "
            f"Details: {status!r}. Please correct the FEN and try again."
        )

    path = find_stockfish_path()
    limit = chess.engine.Limit(depth=depth) if movetime_ms is None else chess.engine.Limit(time=movetime_ms / 1000.0)

    with chess.engine.SimpleEngine.popen_uci(path) as sf:
        infos = sf.analyse(board, limit, multipv=multipv)
        if isinstance(infos, dict):
            infos = [infos]

        def info_to_line(info):
            pv = info.get("pv", [])
            score = info["score"].pov(board.turn)
            if score.is_mate():
                eval_obj = {"type": "mate", "value": score.mate()}
            else:
                eval_obj = {"type": "cp", "value": score.score()}

            san_line = []
            b = board.copy()
            for move in pv:
                san_line.append(b.san(move))
                b.push(move)

            return {
                "eval": eval_obj,
                "best_move_uci": pv[0].uci() if pv else None,
                "best_move_san": san_line[0] if san_line else None,
                "best_line_san": san_line,
                "best_line_uci": [m.uci() for m in pv],
            }

        lines = [info_to_line(info) for info in infos]

    result = {
        "fen": fen,
        "turn": "white" if board.turn == chess.WHITE else "black",
        **lines[0],
    }
    if multipv > 1:
        result["lines"] = lines
    return result
