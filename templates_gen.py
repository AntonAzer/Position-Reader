"""
Generates reference piece images (PNG, transparent background) for all 12
piece types (6 types x 2 colors) using python-chess's built-in SVG piece
renderer (the "cburnett" set used by lichess). These are used as templates
for matching against squares cropped from an uploaded board image.

Run once: python3 templates_gen.py
"""
import os
import chess
import chess.svg
import cairosvg

OUT_DIR = os.path.join(os.path.dirname(__file__), "piece_templates")
SIZE = 200  # render size in pixels (square)

PIECE_TYPES = {
    "P": chess.PAWN,
    "N": chess.KNIGHT,
    "B": chess.BISHOP,
    "R": chess.ROOK,
    "Q": chess.QUEEN,
    "K": chess.KING,
}


def generate():
    os.makedirs(OUT_DIR, exist_ok=True)
    for color_name, color in [("w", chess.WHITE), ("b", chess.BLACK)]:
        for sym, ptype in PIECE_TYPES.items():
            piece = chess.Piece(ptype, color)
            svg_data = chess.svg.piece(piece, size=SIZE)
            out_path = os.path.join(OUT_DIR, f"{color_name}{sym}.png")
            cairosvg.svg2png(
                bytestring=svg_data.encode("utf-8"),
                write_to=out_path,
                output_width=SIZE,
                output_height=SIZE,
                background_color=None,
            )
            print(f"wrote {out_path}")


if __name__ == "__main__":
    generate()
