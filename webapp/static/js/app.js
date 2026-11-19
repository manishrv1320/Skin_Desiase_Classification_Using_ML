(function () {
  "use strict";

  var ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

  var form = document.getElementById("upload-form");
  if (!form) return;

  var input = document.getElementById("file-input");
  var dropzone = document.getElementById("dropzone");
  var prompt = document.getElementById("dropzone-prompt");
  var preview = document.getElementById("preview");
  var fileMeta = document.getElementById("file-meta");
  var submitBtn = document.getElementById("submit-btn");
  var resetBtn = document.getElementById("reset-btn");
  var formError = document.getElementById("form-error");

  var resultsEmpty = document.getElementById("results-empty");
  var results = document.getElementById("results");
  var topName = document.getElementById("top-name");
  var topProb = document.getElementById("top-prob");
  var topSummary = document.getElementById("top-summary");
  var topNote = document.getElementById("top-note");
  var lowConfidence = document.getElementById("low-confidence");
  var ranking = document.getElementById("ranking");

  var maxBytes = Number(form.dataset.maxBytes) || 10 * 1024 * 1024;
  var modelReady = form.dataset.modelReady === "true";
  var endpoint = form.getAttribute("action");

  var currentFile = null;
  var previewUrl = null;

  function formatSize(bytes) {
    if (bytes < 1024 * 1024) return Math.max(1, Math.round(bytes / 1024)) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  function formatPercent(p) {
    var pct = p * 100;
    if (pct > 0 && pct < 0.1) return "<0.1%";
    return pct.toFixed(1) + "%";
  }

  function showError(message) {
    formError.textContent = message;
    formError.hidden = false;
  }

  function clearError() {
    formError.textContent = "";
    formError.hidden = true;
  }

  function clearResults() {
    results.hidden = true;
    resultsEmpty.hidden = false;
    ranking.replaceChildren();
  }

  function setFile(file) {
    clearError();
    if (!file) return;

    if (file.type && ACCEPTED_TYPES.indexOf(file.type) === -1) {
      showError("That file type isn't supported. Use a JPEG, PNG, WebP or BMP image.");
      return;
    }
    if (file.size > maxBytes) {
      showError("This file is " + formatSize(file.size) + ". The limit is " + formatSize(maxBytes) + ".");
      return;
    }

    currentFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);

    preview.src = previewUrl;
    preview.hidden = false;
    prompt.hidden = true;
    dropzone.classList.add("has-image");

    fileMeta.textContent = (file.name || "Pasted image") + " · " + formatSize(file.size);
    fileMeta.hidden = false;

    submitBtn.disabled = !modelReady;
    resetBtn.hidden = false;
    clearResults();
  }

  function reset() {
    currentFile = null;
    input.value = "";
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;

    preview.removeAttribute("src");
    preview.hidden = true;
    prompt.hidden = false;
    dropzone.classList.remove("has-image");
    fileMeta.hidden = true;

    submitBtn.disabled = true;
    resetBtn.hidden = true;
    clearError();
    clearResults();
    input.focus();
  }

  function setBusy(busy) {
    submitBtn.disabled = busy || !currentFile || !modelReady;
    submitBtn.setAttribute("aria-busy", busy ? "true" : "false");
    submitBtn.textContent = busy ? "Running…" : "Run classifier";
    resetBtn.disabled = busy;
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderRankItem(prediction, index) {
    var li = document.createElement("li");
    var details = document.createElement("details");
    var summary = document.createElement("summary");

    summary.appendChild(el("span", "rank", String(index + 1)));
    summary.appendChild(el("span", "rank-name", prediction.label));
    summary.appendChild(el("span", "rank-pct", formatPercent(prediction.probability)));

    var bar = el("span", "rank-bar");
    var fill = document.createElement("span");
    fill.style.width = Math.max(0.5, prediction.probability * 100).toFixed(1) + "%";
    bar.appendChild(fill);
    summary.appendChild(bar);

    details.appendChild(summary);
    details.appendChild(el("p", "rank-detail", prediction.summary || "No description available."));
    li.appendChild(details);
    return li;
  }

  function renderResults(data) {
    var predictions = data.predictions || [];
    if (!predictions.length) {
      showError("The server returned no predictions.");
      return;
    }

    var top = predictions[0];
    topName.textContent = top.label;
    topProb.textContent = formatPercent(top.probability);
    topSummary.textContent = top.summary || "";
    topNote.textContent = top.note || "";
    topNote.hidden = !top.note;
    lowConfidence.hidden = !data.low_confidence;

    ranking.replaceChildren.apply(ranking, predictions.map(renderRankItem));

    resultsEmpty.hidden = true;
    results.hidden = false;

    if (window.matchMedia("(max-width: 820px)").matches) {
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  input.addEventListener("change", function () {
    setFile(input.files && input.files[0]);
  });

  ["dragenter", "dragover"].forEach(function (type) {
    dropzone.addEventListener(type, function (event) {
      event.preventDefault();
      dropzone.classList.add("is-dragging");
    });
  });

  ["dragleave", "dragend", "drop"].forEach(function (type) {
    dropzone.addEventListener(type, function () {
      dropzone.classList.remove("is-dragging");
    });
  });

  dropzone.addEventListener("drop", function (event) {
    event.preventDefault();
    var files = event.dataTransfer && event.dataTransfer.files;
    if (files && files.length) setFile(files[0]);
  });

  document.addEventListener("paste", function (event) {
    var items = (event.clipboardData && event.clipboardData.items) || [];
    for (var i = 0; i < items.length; i++) {
      if (items[i].kind === "file" && items[i].type.indexOf("image/") === 0) {
        setFile(items[i].getAsFile());
        event.preventDefault();
        return;
      }
    }
  });

  resetBtn.addEventListener("click", reset);

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    if (!currentFile || !modelReady) return;

    clearError();
    setBusy(true);

    var body = new FormData();
    body.append("image", currentFile, currentFile.name || "upload");

    fetch(endpoint, { method: "POST", body: body })
      .catch(function () {
        throw new Error("Couldn't reach the server. Check that it is still running and try again.");
      })
      .then(function (response) {
        return response
          .json()
          .catch(function () {
            return {};
          })
          .then(function (data) {
            if (!response.ok) {
              throw new Error(data.error || "The server returned an error (" + response.status + ").");
            }
            return data;
          });
      })
      .then(renderResults)
      .catch(function (err) {
        showError(err.message);
      })
      .finally(function () {
        setBusy(false);
      });
  });
})();
