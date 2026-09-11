(() => {
  const dropzone = document.getElementById("dropzone");
  const dropzoneContent = document.getElementById("dropzoneContent");
  const fileInput = document.getElementById("fileInput");
  const previewImg = document.getElementById("previewImg");
  const recognizeBtn = document.getElementById("recognizeBtn");
  const recognizeStatus = document.getElementById("recognizeStatus");

  const turnSelect = document.getElementById("turnSelect");

  const boardSvg = document.getElementById("boardSvg");
  const boardPlaceholder = document.getElementById("boardPlaceholder");
  const fenInput = document.getElementById("fenInput");
  const confidenceNote = document.getElementById("confidenceNote");

  const depthRange = document.getElementById("depthRange");
  const depthValue = document.getElementById("depthValue");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const analyzeStatus = document.getElementById("analyzeStatus");
  const resultBlock = document.getElementById("resultBlock");
  const evalValue = document.getElementById("evalValue");
  const evalWho = document.getElementById("evalWho");
  const lineList = document.getElementById("lineList");

  let currentTurn = "w";
  let selectedFile = null;

  // ---------- Upload / dropzone ----------

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      setStatus(recognizeStatus, "That doesn't look like an image file.", "error");
      return;
    }
    selectedFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewImg.hidden = false;
    dropzoneContent.hidden = true;
    recognizeBtn.disabled = false;
    setStatus(recognizeStatus, "", "");
  }

  // ---------- Turn toggle ----------

  turnSelect.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    [...turnSelect.querySelectorAll(".seg-btn")].forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentTurn = btn.dataset.turn;
  });

  // ---------- Recognize ----------

  recognizeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    recognizeBtn.disabled = true;
    setStatus(recognizeStatus, "Reading the board…", "");

    const form = new FormData();
    form.append("image", selectedFile);
    form.append("turn", currentTurn);

    try {
      const res = await fetch("/api/recognize", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Recognition failed.");

      fenInput.value = data.fen;
      renderBoard(data.fen);

      const confPct = Math.round(data.avg_confidence * 100);
      let note = `Detected with ~${confPct}% average confidence.`;
      let cls = "";
      if (data.low_confidence_squares && data.low_confidence_squares.length) {
        note += ` Double-check: ${data.low_confidence_squares.join(", ")}.`;
        cls = "warn";
      }
      if (!data.board_detected_via_contour) {
        note += " Assumed the whole image is the board.";
      }
      confidenceNote.textContent = note;
      confidenceNote.className = "confidence-note " + cls;

      setStatus(recognizeStatus, "Done. Review the FEN, then find the best line.", "ok");
    } catch (err) {
      setStatus(recognizeStatus, err.message, "error");
    } finally {
      recognizeBtn.disabled = false;
    }
  });

  // ---------- FEN -> board preview ----------

  fenInput.addEventListener("input", () => {
    const fen = fenInput.value.trim();
    if (fen) renderBoard(fen);
  });

  function renderBoard(fen) {
    boardSvg.src = "/api/board.svg?fen=" + encodeURIComponent(fen) + "&_=" + Date.now();
    boardSvg.hidden = false;
    boardPlaceholder.hidden = true;
  }

  // ---------- Depth slider ----------

  depthRange.addEventListener("input", () => {
    depthValue.textContent = depthRange.value;
  });

  // ---------- Analyze ----------

  analyzeBtn.addEventListener("click", async () => {
    const fen = fenInput.value.trim();
    if (!fen) {
      setStatus(analyzeStatus, "Recognize an image or enter a FEN first.", "error");
      return;
    }
    analyzeBtn.disabled = true;
    resultBlock.hidden = true;
    setStatus(analyzeStatus, "Thinking…", "");

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen, depth: Number(depthRange.value), multipv: 1 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Analysis failed.");

      showResult(data);
      setStatus(analyzeStatus, "", "");
    } catch (err) {
      setStatus(analyzeStatus, err.message, "error");
    } finally {
      analyzeBtn.disabled = false;
    }
  });

  function showResult(data) {
    resultBlock.hidden = false;

    if (data.eval.type === "mate") {
      const n = data.eval.value;
      evalValue.textContent = (n > 0 ? "#" : "#-") + Math.abs(n);
    } else {
      const cp = data.eval.value;
      const pawns = (cp / 100).toFixed(2);
      evalValue.textContent = (cp > 0 ? "+" : "") + pawns;
    }
    evalWho.textContent = `for ${data.turn} to move`;

    lineList.innerHTML = "";
    const line = data.best_line_san || [];
    let moveNo = data.turn === "black"
      ? (data.fullmove || 1)
      : (data.fullmove || 1);

    // Reconstruct move numbering from the FEN's fullmove counter if present.
    const fenParts = data.fen.split(" ");
    let fullmove = parseInt(fenParts[5], 10) || 1;
    let sideIsWhite = data.turn === "white";

    for (let i = 0; i < line.length; i++) {
      const li = document.createElement("li");
      const isFirst = i === 0;
      const numberLabel = sideIsWhite ? `${fullmove}.` : (i === 0 ? `${fullmove}…` : `${fullmove}.`);

      const noSpan = document.createElement("span");
      noSpan.className = "move-no";
      noSpan.textContent = numberLabel;

      const sanSpan = document.createElement("span");
      sanSpan.className = "san" + (isFirst ? " first-move" : "");
      sanSpan.textContent = line[i];

      li.appendChild(noSpan);
      li.appendChild(sanSpan);
      lineList.appendChild(li);

      if (sideIsWhite) {
        sideIsWhite = false;
      } else {
        sideIsWhite = true;
        fullmove += 1;
      }
    }
  }

  function setStatus(el, text, kind) {
    el.textContent = text;
    el.className = "status-line" + (kind ? " " + kind : "");
  }
})();
