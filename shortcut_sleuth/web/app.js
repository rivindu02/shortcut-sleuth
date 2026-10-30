"use strict";
const state = {shape: 0, background: 0, seed: 73, catalog: null, example: null, request: 0, selected: {}};
const names = ["Circle", "Square", "Triangle"];
const $ = id => document.getElementById(id);

async function json(url) {
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function modelCard(architecture) {
  const condition = $("condition").value;
  const models = state.catalog.models.filter(m => m.architecture === architecture && m.condition === condition);
  const card = element("article", "panel model-card");
  const title = element("div", "model-title");
  title.append(element("h3", "", architecture === "cnn" ? "Small CNN" : "Small ViT"), element("span", "", "FROM SCRATCH"));
  card.append(title);
  if (!models.length) {
    const empty = element("div", "empty-state");
    empty.append(element("strong", "", "Awaiting a checkpoint"), element("p", "", "Train the smoke matrix and restart the demo with --runs to inspect real predictions. No model scores are shown before training."));
    card.append(empty);
    return card;
  }
  let model = models.find(m => m.id === state.selected[architecture]) || models[0];
  state.selected[architecture] = model.id;
  const choice = element("select", "model-choice");
  choice.setAttribute("aria-label", `${architecture.toUpperCase()} checkpoint`);
  models.forEach(m => {
    const option = element("option", "", `Seed ${m.seed} · ${m.protocol} · ${m.parameters.toLocaleString()} parameters`);
    option.value = m.id;
    choice.append(option);
  });
  choice.value = model.id;
  choice.addEventListener("change", () => {state.selected[architecture] = choice.value; render();});
  card.append(choice);
  if (!state.example) {
    card.append(element("p", "model-note", "Computing predictions for the paired object…"));
    return card;
  }
  const paired = state.example.predictions[model.id];
  const p = paired[state.background];
  const predictions = paired.map(row => row.indexOf(Math.max(...row)));
  const predicted = predictions[state.background];
  card.append(element("div", "model-result", names[predicted]));
  p.forEach((value, i) => {
    const row = element("div", "probability-row");
    const track = element("div", "track");
    // Use a progress element so the CSP can prohibit inline style attributes.
    const progress = document.createElement("progress");
    progress.max = 1;
    progress.value = value;
    progress.setAttribute("aria-label", `${names[i]} probability`);
    track.append(progress);
    row.append(element("span", "", names[i]), track, element("span", "", `${(100 * value).toFixed(1)}%`));
    card.append(row);
  });
  const flip = new Set(predictions).size > 1;
  const trueRange = Math.max(...paired.map(row => row[state.shape])) - Math.min(...paired.map(row => row[state.shape]));
  card.append(element("p", "model-note", `${flip ? "Prediction changes" : "Prediction stable"} across backgrounds · true-class range ${(100 * trueRange).toFixed(1)} pp`));
  return card;
}

function render() {
  $("specimen-image").hidden = !state.example;
  document.querySelectorAll("[data-shape]").forEach(b => b.setAttribute("aria-pressed", String(Number(b.dataset.shape) === state.shape)));
  document.querySelectorAll("[data-background]").forEach(b => b.setAttribute("aria-pressed", String(Number(b.dataset.background) === state.background)));
  if (state.example) {
    $("specimen-image").src = state.example.images[state.background];
    $("specimen-image").alt = `${names[state.shape]} on ${state.catalog.backgrounds[state.background]} background`;
    $("object-id").textContent = `${state.example.object.object_id} · mask ${state.example.foreground_hash.slice(0, 16)}`;
    $("paired-images").replaceChildren(...state.example.images.map((src, b) => {
      const button = element("button");
      button.type = "button";
      button.setAttribute("aria-label", `Select ${state.catalog.backgrounds[b]} background`);
      button.setAttribute("aria-pressed", String(b === state.background));
      const img = element("img");
      img.src = src;
      img.alt = `${names[state.shape]} on ${state.catalog.backgrounds[b]}`;
      button.append(img);
      button.addEventListener("click", () => {state.background = b; render();});
      return button;
    }));
  } else {
    $("paired-images").replaceChildren();
    $("object-id").textContent = "Rendering paired object…";
  }
  $("baseline-prediction").textContent = `Background says: ${names[state.background]}`;
  $("model-cards").replaceChildren(modelCard("cnn"), modelCard("vit"));
}

async function loadExample() {
  const request = ++state.request;
  state.example = null;
  render();
  $("specimen-image").setAttribute("aria-busy", "true");
  $("error").hidden = true;
  $("new-object").disabled = true;
  try {
    const example = await json(`/api/example?shape=${state.shape}&seed=${state.seed}`);
    if (request !== state.request) return;
    state.example = example;
    render();
  } catch (error) {
    if (request !== state.request) return;
    $("error").textContent = error.message;
    $("error").hidden = false;
  } finally {
    if (request === state.request) {
      $("new-object").disabled = false;
      $("specimen-image").setAttribute("aria-busy", "false");
    }
  }
}

async function start() {
  state.catalog = await json("/api/catalog");
  $("mode-banner").textContent = state.catalog.models.length
    ? `${state.catalog.models.length} real checkpoints loaded. Predictions below are computed from the displayed objects. Check each checkpoint’s protocol and seed; smoke runs are development evidence.`
    : "Renderer + analytic control mode. No trained checkpoints are loaded. Explore exact background swaps below; CNN and ViT predictions appear after training.";
  $("condition").disabled = !state.catalog.models.length;
  document.querySelectorAll("[data-shape]").forEach(button => button.addEventListener("click", () => {
    state.shape = Number(button.dataset.shape);
    loadExample();
  }));
  document.querySelectorAll("[data-background]").forEach(button => button.addEventListener("click", () => {
    state.background = Number(button.dataset.background);
    render();
  }));
  $("new-object").addEventListener("click", () => {
    state.seed = (state.seed + 1) >>> 0;
    $("object-seed").value = state.seed;
    loadExample();
  });
  $("object-seed").addEventListener("change", () => {
    if (!$("object-seed").checkValidity() || $("object-seed").value === "") {
      $("object-seed").reportValidity();
      return;
    }
    state.seed = Number($("object-seed").value);
    loadExample();
  });
  $("condition").addEventListener("change", render);
  await loadExample();
}

start().catch(error => {
  $("mode-banner").textContent = "Cannot connect to the local experiment server.";
  $("error").textContent = error.message;
  $("error").hidden = false;
});
