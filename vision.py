"""
Chess board vision pipeline.

Given an uploaded image of a chessboard (ideally a clean digital screenshot,
e.g. from lichess/chess.com), this module:
  1. Locates the board region in the image.
  2. Splits it into a fixed grid of 8x8 squares.
  3. For each square, determines whether it holds a piece and, if so,
     classifies its type and color via silhouette template matching against
     reference piece renders (see templates_gen.py).

This is a classical-CV approach (no training data / GPU required), so it
works best on clean, mostly-frontal digital board screenshots. Real photos
of a physical board with glare/angle/depth will be considerably less
reliable -- the confidence score returned is meant to help flag that.
"""
import os
import numpy as np
import cv2

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "piece_templates")
CANVAS = 200  # must match templates_gen.SIZE

PIECE_FILES = {
    "wP": "wP.png", "wN": "wN.png", "wB": "wB.png", "wR": "wR.png", "wQ": "wQ.png", "wK": "wK.png",
    "bP": "bP.png", "bN": "bN.png", "bB": "bB.png", "bR": "bR.png", "bQ": "bQ.png", "bK": "bK.png",
}

FEN_LETTER = {
    "P": "P", "N": "N", "B": "B", "R": "R", "Q": "Q", "K": "K",
}


def _normalize_mask(mask, contour):
    """Crop a mask to its piece bounding box and center it, filling 90% of
    a CANVAS x CANVAS square. Used for BOTH templates and extracted cell
    silhouettes so the two are compared on an apples-to-apples basis
    (otherwise leftover SVG margins in the templates skew matching)."""
    x, y, w, h = cv2.boundingRect(contour)
    crop = mask[y:y + h, x:x + w]
    scale = (CANVAS * 0.9) / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((CANVAS, CANVAS), np.uint8)
    ox, oy = (CANVAS - nw) // 2, (CANVAS - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _load_templates():
    """Load reference piece silhouettes (binary masks) keyed by e.g. 'wQ',
    normalized (cropped to bbox + re-centered/scaled) so they're directly
    comparable to normalized cell silhouettes."""
    templates = {}
    for key, fname in PIECE_FILES.items():
        path = os.path.join(TEMPLATE_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGRA
        alpha = img[:, :, 3]
        mask = (alpha > 20).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Use the union of all contours' bounding box (pieces like the king
        # have a small disconnected cross on top).
        all_pts = np.concatenate(contours, axis=0)
        x, y, w, h = cv2.boundingRect(all_pts)
        crop = mask[y:y + h, x:x + w]
        scale = (CANVAS * 0.9) / max(w, h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((CANVAS, CANVAS), np.uint8)
        ox, oy = (CANVAS - nw) // 2, (CANVAS - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        templates[key] = canvas
    return templates


_TEMPLATES = _load_templates()


def _order_quad_points(pts):
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]       # top-left
    ordered[2] = pts[np.argmax(s)]       # bottom-right
    ordered[1] = pts[np.argmin(diff)]    # top-right
    ordered[3] = pts[np.argmax(diff)]    # bottom-left
    return ordered


def find_board(img_bgr, out_size=800):
    """
    Try to locate a square board contour in the image and warp it to a
    flat out_size x out_size image. Falls back to just resizing the whole
    image if no confident board contour is found (common & fine for
    screenshots that are already tightly cropped to the board).

    Returns (warped_board_bgr, board_was_detected: bool)
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = h * w
    best = None
    best_area = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            # Require a substantial, near-square, axis-aligned quad. This
            # avoids false positives from internal shapes (e.g. pieces
            # whose outlines happen to approximate a skewed quad) and only
            # fires for a genuine board-within-a-larger-photo case.
            if area < 0.5 * img_area:
                continue
            x, y, bw, bh = cv2.boundingRect(approx)
            aspect = bw / float(bh)
            if not (0.9 < aspect < 1.11):
                continue

            pts = _order_quad_points(approx.astype(np.float32))
            side_lengths = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
            if max(side_lengths) / min(side_lengths) > 1.12:
                continue  # not square enough

            # Check axis alignment: each side should be close to horizontal
            # or vertical (near-frontal screenshot/photo), within ~8 degrees.
            axis_aligned = True
            for i in range(4):
                dx, dy = pts[(i + 1) % 4] - pts[i]
                angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
                if not (angle < 4 or angle > 86):
                    axis_aligned = False
                    break
            if not axis_aligned:
                continue

            if area > best_area:
                best = approx
                best_area = area

    if best is not None:
        ordered = _order_quad_points(best.astype(np.float32))
        dst = np.array(
            [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(img_bgr, M, (out_size, out_size))
        return warped, True

    # Fallback: assume the whole image is (approximately) the board.
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = img_bgr[y0:y0 + side, x0:x0 + side]
    resized = cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return resized, False


def _extract_silhouette(cell_bgr):
    """
    Estimate the background (square) color from the cell corners, then
    threshold the rest of the cell against that background to get a
    foreground (piece) mask. Returns (mask_uint8, foreground_fraction).
    """
    h, w = cell_bgr.shape[:2]
    m = max(2, int(min(h, w) * 0.08))
    corner_patches = [
        cell_bgr[0:m, 0:m], cell_bgr[0:m, w - m:w],
        cell_bgr[h - m:h, 0:m], cell_bgr[h - m:h, w - m:w],
    ]
    bg = np.median(np.concatenate([p.reshape(-1, 3) for p in corner_patches], axis=0), axis=0)

    diff = np.linalg.norm(cell_bgr.astype(np.float32) - bg.astype(np.float32), axis=2)
    thresh = max(28.0, float(np.percentile(diff, 85)) * 0.35)
    mask = (diff > thresh).astype(np.uint8) * 255

    # Trim a thin border (anti-aliased square edges / grid lines) to avoid
    # false foreground from board borders rather than a piece.
    border = max(1, int(min(h, w) * 0.03))
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), np.uint8), 0.0, None

    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
    frac = cv2.countNonZero(clean_mask) / float(h * w)
    return clean_mask, frac, largest


def _mask_to_canvas(mask, contour):
    """Crop mask to its bounding box and center it on a CANVAS x CANVAS square."""
    x, y, w, h = cv2.boundingRect(contour)
    crop = mask[y:y + h, x:x + w]
    scale = (CANVAS * 0.9) / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CANVAS, CANVAS), np.uint8)
    ox, oy = (CANVAS - nw) // 2, (CANVAS - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _best_template_match(canvas_mask):
    """Compare a normalized silhouette against all 12 piece templates (shape only)."""
    best_key, best_score = None, -1.0
    for key, tmpl in _TEMPLATES.items():
        inter = cv2.bitwise_and(canvas_mask, tmpl)
        union = cv2.bitwise_or(canvas_mask, tmpl)
        iou = cv2.countNonZero(inter) / float(max(1, cv2.countNonZero(union)))
        if iou > best_score:
            best_score = iou
            best_key = key
    return best_key, best_score


def _piece_color(cell_bgr, mask):
    """
    Determine white vs black piece by looking at the brightness of the
    original-image pixels *inside* the silhouette, focusing on the most
    extreme (brightest and darkest) pixels -- pieces are typically a
    solid white or solid black fill with a contrasting outline, so the
    fill dominates either the bright or dark extreme, regardless of the
    square's own background color.
    Returns ('w' or 'b', confidence 0..1).
    """
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    pixels = gray[mask > 0]
    if pixels.size == 0:
        return "w", 0.0
    # Erode the mask slightly so we sample interior fill, not the outline.
    eroded = cv2.erode(mask, np.ones((5, 5), np.uint8))
    interior = gray[eroded > 0]
    sample = interior if interior.size > 20 else pixels

    dark_frac = float((sample < 90).sum()) / sample.size
    light_frac = float((sample > 165).sum()) / sample.size

    if light_frac >= dark_frac:
        return "w", light_frac
    return "b", dark_frac


def classify_cell(cell_bgr):
    """
    Returns (fen_symbol_or_None, confidence 0..1).
    fen_symbol uses standard FEN letters, uppercase = white, lowercase = black.
    None means the square is judged empty.
    """
    mask, frac, contour = _extract_silhouette(cell_bgr)
    if frac < 0.035 or contour is None:
        return None, 1.0 - frac  # confident "empty" when foreground fraction is tiny

    canvas = _mask_to_canvas(mask, contour)
    key, shape_score = _best_template_match(canvas)  # e.g. 'wQ' (color part unused)
    letter = key[1]
    color, color_score = _piece_color(cell_bgr, mask)
    symbol = letter if color == "w" else letter.lower()
    confidence = 0.6 * shape_score + 0.4 * color_score
    return symbol, confidence


def image_to_grid(img_bgr):
    """
    Full pipeline: image -> 8x8 grid (list of 8 lists of 8), rank 8 (top)
    to rank 1 (bottom), each cell a FEN symbol or None. Also returns a
    per-cell confidence grid and whether the board was auto-detected.
    """
    board_img, detected = find_board(img_bgr, out_size=800)
    size = board_img.shape[0]
    step = size // 8

    grid = [[None] * 8 for _ in range(8)]
    conf = [[0.0] * 8 for _ in range(8)]

    for row in range(8):          # row 0 = rank 8 (top of image)
        for col in range(8):      # col 0 = file a (left of image)
            y0, y1 = row * step, (row + 1) * step
            x0, x1 = col * step, (col + 1) * step
            cell = board_img[y0:y1, x0:x1]
            symbol, score = classify_cell(cell)
            grid[row][col] = symbol
            conf[row][col] = score

    avg_conf = float(np.mean(conf))
    return grid, conf, avg_conf, detected


def grid_to_fen(grid, turn="w", castling="KQkq", ep="-", halfmove=0, fullmove=1):
    """grid[0] = rank 8 ... grid[7] = rank 1, each row left(file a)->right(file h)."""
    rows = []
    for row in grid:
        fen_row = ""
        empty = 0
        for symbol in row:
            if symbol is None:
                empty += 1
            else:
                if empty:
                    fen_row += str(empty)
                    empty = 0
                fen_row += symbol
        if empty:
            fen_row += str(empty)
        rows.append(fen_row)
    board_part = "/".join(rows)
    return f"{board_part} {turn} {castling} {ep} {halfmove} {fullmove}"


def infer_castling_rights(grid):
    """
    Best-effort castling rights from *piece placement alone*: a right is
    only plausible if the relevant king and rook are still on their home
    squares. This can't know whether a king already moved and moved back,
    but it's a far better default than always assuming full rights, and
    the user can hand-edit the resulting FEN if needed.
    grid[0] = rank 8 ... grid[7] = rank 1, row[0] = file a ... row[7] = file h.
    """
    rank1, rank8 = grid[7], grid[0]
    rights = ""
    if rank1[4] == "K":
        if rank1[7] == "R":
            rights += "K"
        if rank1[0] == "R":
            rights += "Q"
    if rank8[4] == "k":
        if rank8[7] == "r":
            rights += "k"
        if rank8[0] == "r":
            rights += "q"
    return rights or "-"
