import io
import traceback

import cv2
import numpy as np
import chess
import chess.svg
from flask import Flask, request, jsonify, render_template, Response

from vision import image_to_grid, grid_to_fen, infer_castling_rights
from engine import analyze_fen

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB upload limit


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded (field name 'image')."}), 400

    file = request.files["image"]
    turn = request.form.get("turn", "w").strip().lower()
    turn = "w" if turn.startswith("w") else "b"

    data = file.read()
    if not data:
        return jsonify({"error": "Uploaded file is empty."}), 400

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image. Please upload a PNG/JPG."}), 400

    try:
        grid, conf, avg_conf, detected = image_to_grid(img)
        castling = infer_castling_rights(grid)
        fen = grid_to_fen(grid, turn=turn, castling=castling)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Vision pipeline failed: {e}"}), 500

    # Low-confidence squares, useful for the frontend to highlight for the
    # user to double check / correct.
    low_conf_squares = []
    files = "abcdefgh"
    for row in range(8):
        rank = 8 - row
        for col in range(8):
            if conf[row][col] < 0.75:
                low_conf_squares.append(f"{files[col]}{rank}")

    return jsonify({
        "fen": fen,
        "board_detected_via_contour": detected,
        "avg_confidence": round(avg_conf, 3),
        "low_confidence_squares": low_conf_squares,
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(silent=True) or {}
    fen = (body.get("fen") or "").strip()
    if not fen:
        return jsonify({"error": "Missing 'fen'."}), 400

    depth = int(body.get("depth", 16))
    depth = max(6, min(depth, 22))
    multipv = int(body.get("multipv", 3))
    multipv = max(1, min(multipv, 5))

    try:
        result = analyze_fen(fen, depth=depth, multipv=multipv)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Engine analysis failed: {e}"}), 500

    return jsonify(result)


@app.route("/api/board.svg")
def board_svg():
    fen = request.args.get("fen", "").strip()
    last_move_uci = request.args.get("lastmove", "").strip()
    try:
        board = chess.Board(fen) if fen else chess.Board.empty()
    except ValueError:
        board = chess.Board.empty()

    lastmove = None
    if last_move_uci:
        try:
            lastmove = chess.Move.from_uci(last_move_uci)
        except ValueError:
            lastmove = None

    svg_data = chess.svg.board(
        board,
        size=420,
        coordinates=True,
        lastmove=lastmove,
        colors={
            "square light": "#EDE6D6",
            "square dark": "#5C6E52",
            "square light lastmove": "#E8D9A0",
            "square dark lastmove": "#BFA855",
            "coord": "#8B8378",
        },
    )
    return Response(svg_data, mimetype="image/svg+xml")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
