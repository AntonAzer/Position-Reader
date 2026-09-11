# Position Reader

Upload an image of a chess board, get the position as FEN, and get Stockfish's
best line from there — as a small local web app.

## How it works

1. **Vision** (`vision.py`) — locates the board in the image, splits it into
   an 8x8 grid, and classifies each square by comparing its piece silhouette
   against reference piece shapes (rendered from the standard "cburnett" set
   via `python-chess`). Works best on **clean digital screenshots**
   (lichess, chess.com, etc.) — a classical computer-vision approach like
   this is much less reliable on photos of a physical board (glare, angle,
   depth), since it has no learned model and no training data behind it.
2. **Turn / castling** — you tell it whose move it is (there's no reliable
   way to read that from a single static image). Castling rights are
   inferred from whether each king/rook is still on its home square.
3. **Engine** (`engine.py`) — runs the real Stockfish binary via
   `python-chess` and returns the evaluation and best line (principal
   variation) from the position.
4. **Web app** (`app.py` + `templates/` + `static/`) — a small Flask app
   tying it together: upload → review/correct the FEN → get the line.

## Setup

### 1. Install Stockfish

```bash
# Debian/Ubuntu
sudo apt-get install stockfish

# macOS
brew install stockfish

# Or download a binary from https://stockfishchess.org/download/
```

The app looks for the binary at `/usr/games/stockfish`, `/usr/bin/stockfish`,
or on your `PATH`. If yours lives elsewhere, edit `STOCKFISH_CANDIDATES` at
the top of `engine.py`.

### 2. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Generate the piece reference templates (already included, but if you
   ever need to regenerate them, e.g. after editing `templates_gen.py`):

```bash
python3 templates_gen.py
```

### 4. Run it

```bash
python3 app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Using it

1. Drop in a board image (a screenshot works best).
2. Pick who's to move.
3. Click **Read the position** — it'll show you the detected FEN and a
   clean re-drawn board so you can visually double-check it. Low-confidence
   squares are called out — fix the FEN text directly if anything's wrong.
4. Adjust the engine depth if you like (higher = stronger but slower).
5. Click **Find the best line**.

You can also skip the image step entirely and just paste a FEN straight
into the FEN box.

## Extending this

- **Better vision / physical board photos**: the current approach is
  classical CV (silhouette + template matching), not a trained model. For
  real photos, you'd want a proper CNN piece classifier (needs labeled
  training data) plus more robust board-corner detection (e.g. a
  keypoint/homography model) to handle perspective, glare, and shadows.
- **Multi-line analysis**: `engine.py`'s `analyze_fen(..., multipv=3)`
  already supports returning multiple candidate lines — the frontend only
  requests `multipv=1` right now but the API can return more.
- **Manual board correction UI**: right now corrections happen by hand-editing
  the FEN text. A clickable grid to fix individual squares would be a nice
  follow-up.

## Project structure

```
app.py                 Flask server & API routes
vision.py               Image -> 8x8 grid -> FEN
engine.py                Stockfish wrapper
templates_gen.py          Generates piece_templates/*.png (run once, already done)
piece_templates/            Reference piece silhouettes used for matching
templates/index.html          Page markup
static/style.css               Styling
static/app.js                   Frontend logic
requirements.txt
```
