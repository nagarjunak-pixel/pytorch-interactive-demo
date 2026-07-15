// Interactive PyTorch Concepts Demo - App Coordinator
// Coordinate UI events, state, canvas render loops, and tabs

import { embed, cosineSimilarity } from "./src/embeddings.js";
import { SimpleNeuralNet, generateDataset } from "./src/nn.js";
import { drawEmbeddingSpace, drawFunctionFit, drawLossCurve } from "./src/visualizer.js";

// --- State Variables ---
const DEFAULT_DOCUMENTS = [
  "Deep learning is a subset of machine learning.",
  "PyTorch is an open-source machine learning library.",
  "Natural language processing handles human text.",
  "Vector databases store embeddings for RAG retrieval.",
  "Cosine similarity measures the angle between vectors.",
  "Pinecone, Weaviate, and pgvector are vector databases."
];

let documents = [...DEFAULT_DOCUMENTS];
let activeTab = "tab-intro";

// Neural Network training state
let nnNet = null;
let nnDataset = [];
let nnLossHistory = [];
let nnEpoch = 0;
let nnAutoplayId = null;
let nnLr = 0.05;

// --- DOM References ---
const tabButtons = document.querySelectorAll(".tab-btn");
const tabPanels = document.querySelectorAll(".tab-panel");

// Embeddings Panel DOM
const documentInputsList = document.getElementById("document-inputs-list");
const btnResetDocs = document.getElementById("btn-reset-docs");
const queryInput = document.getElementById("query-input");
const btnSearch = document.getElementById("btn-search");
const queryVectorCells = document.getElementById("query-vector-cells");
const retrievalResults = document.getElementById("retrieval-results");
const embeddingsCanvas = document.getElementById("embeddings-canvas");

// Neural Network Panel DOM
const nnTaskSelect = document.getElementById("nn-task-select");
const nnLrSlider = document.getElementById("nn-lr-slider");
const nnLrValue = document.getElementById("nn-lr-value");
const btnTrain1 = document.getElementById("btn-train-1");
const btnTrain50 = document.getElementById("btn-train-50");
const btnTrainAuto = document.getElementById("btn-train-auto");
const btnNnReset = document.getElementById("btn-nn-reset");
const statEpoch = document.getElementById("stat-epoch");
const statLoss = document.getElementById("stat-loss");
const weightLayer1Grid = document.getElementById("weight-layer1-grid");
const weightLayer2Grid = document.getElementById("weight-layer2-grid");
const weightLayer1Title = document.getElementById("weight-layer1-title");
const weightLayer2Title = document.getElementById("weight-layer2-title");
const nnPlotCanvas = document.getElementById("nn-plot-canvas");
const nnLossCanvas = document.getElementById("nn-loss-canvas");
const nnPlotTitle = document.getElementById("nn-plot-title");
const nnPlotCaption = document.getElementById("nn-plot-caption");

// --- Initialization ---
function init() {
  setupTabs();
  setupEmbeddingsPanel();
  setupNNPanel();
  
  // Perform initial search computation
  runSearch();
}

// --- Tabs Logic ---
function setupTabs() {
  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      
      // Update nav class
      tabButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      // Update panel class
      tabPanels.forEach(panel => {
        panel.classList.remove("active");
        if (panel.id === targetTab) {
          panel.classList.add("active");
        }
      });

      activeTab = targetTab;
      
      // Stop training autoplay if user leaves NN tab
      if (activeTab !== "tab-nn" && nnAutoplayId !== null) {
        toggleNNAutoplay();
      }

      // Redraw canvases when switching to respective tabs to guarantee rendering
      if (activeTab === "tab-embeddings") {
        runSearch();
      } else if (activeTab === "tab-nn") {
        updateNNPlots();
      }
    });
  });
}

// --- Concept A: Embeddings & Retrieval Logic ---
function setupEmbeddingsPanel() {
  renderDocumentInputs();

  btnResetDocs.addEventListener("click", () => {
    documents = [...DEFAULT_DOCUMENTS];
    renderDocumentInputs();
    runSearch();
  });

  btnSearch.addEventListener("click", runSearch);
  queryInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") runSearch();
  });
}

function renderDocumentInputs() {
  documentInputsList.innerHTML = "";
  documents.forEach((docText, idx) => {
    const docDiv = document.createElement("div");
    docDiv.className = "doc-item";

    const numSpan = document.createElement("span");
    numSpan.className = "doc-number";
    numSpan.textContent = idx + 1;

    const input = document.createElement("input");
    input.type = "text";
    input.value = docText;
    input.addEventListener("input", (e) => {
      documents[idx] = e.target.value;
      runSearch(); // Reactive update
    });

    docDiv.appendChild(numSpan);
    docDiv.appendChild(input);
    documentInputsList.appendChild(docDiv);
  });
}

function runSearch() {
  const queryText = queryInput.value.trim();
  const queryVec = embed(queryText);

  // Update query vector representation in UI
  queryVectorCells.innerHTML = "";
  queryVec.forEach(val => {
    const cell = document.createElement("div");
    cell.className = "vector-cell";
    cell.textContent = val.toFixed(2);
    // Dynamic color highlight intensity
    const intensity = Math.round(val * 100);
    cell.style.background = `rgba(20, 184, 166, ${Math.min(0.25, val * 0.25)})`;
    cell.style.borderColor = `rgba(20, 184, 166, ${Math.min(0.5, val * 0.5)})`;
    queryVectorCells.appendChild(cell);
  });

  // Embed documents and calculate similarity
  const scoredDocs = documents.map((text, idx) => {
    const vec = embed(text);
    const sim = cosineSimilarity(queryVec, vec);
    return {
      index: idx,
      text: text,
      vector: vec,
      score: sim
    };
  });

  // Sort documents by similarity score
  const rankedDocs = [...scoredDocs].sort((a, b) => b.score - a.score);

  // Render ranking list in UI
  retrievalResults.innerHTML = "";
  
  // Identify top-3 index numbers
  const top3Indexes = rankedDocs.slice(0, 3).map(d => d.index);

  rankedDocs.forEach((doc, rank) => {
    const isTop3 = top3Indexes.includes(doc.index);
    const scorePercent = Math.max(0, Math.round(doc.score * 100));

    const item = document.createElement("div");
    item.className = `ranking-item ${isTop3 ? "top-match" : ""}`;

    const header = document.createElement("div");
    header.className = "ranking-header";

    const textDiv = document.createElement("div");
    textDiv.className = "ranking-text";
    textDiv.innerHTML = `<span style="color: var(--text-dark); font-weight:700;">#${doc.index + 1}</span> ${doc.text}`;

    const meta = document.createElement("div");
    meta.className = "ranking-meta";

    const badge = document.createElement("span");
    badge.className = `ranking-score ${doc.score > 0.6 ? "score-high" : ""}`;
    badge.textContent = `cos_sim = ${doc.score.toFixed(3)}`;

    meta.appendChild(badge);
    header.appendChild(textDiv);
    header.appendChild(meta);

    const barOuter = document.createElement("div");
    barOuter.className = "ranking-bar-outer";

    const barInner = document.createElement("div");
    barInner.className = "ranking-bar-inner";
    // Scale cosine score visual range for better contrast: [0, 1] maps to bar width
    barInner.style.width = `${Math.max(0, doc.score) * 100}%`;

    barOuter.appendChild(barInner);
    item.appendChild(header);
    item.appendChild(barOuter);

    retrievalResults.appendChild(item);
  });

  // Render radar spatial canvas
  drawEmbeddingSpace(embeddingsCanvas, scoredDocs, { vector: queryVec }, top3Indexes);
}


// --- Concept B: Neural Network Logic ---
function setupNNPanel() {
  recreateNN();

  nnTaskSelect.addEventListener("change", recreateNN);
  
  nnLrSlider.addEventListener("input", (e) => {
    nnLr = parseFloat(e.target.value);
    nnLrValue.textContent = nnLr.toFixed(2);
  });

  btnTrain1.addEventListener("click", () => {
    trainSteps(1);
  });

  btnTrain50.addEventListener("click", () => {
    trainSteps(50);
  });

  btnTrainAuto.addEventListener("click", toggleNNAutoplay);
  
  btnNnReset.addEventListener("click", () => {
    if (nnAutoplayId !== null) toggleNNAutoplay();
    nnNet.reset();
    nnLossHistory = [];
    nnEpoch = 0;
    updateNNStats();
    updateNNPlots();
    renderNNParameters();
  });
}

function recreateNN() {
  if (nnAutoplayId !== null) toggleNNAutoplay();

  const mode = nnTaskSelect.value;
  nnLossHistory = [];
  nnEpoch = 0;

  if (mode === "regression") {
    nnNet = new SimpleNeuralNet(1, 6, 1, "regression");
    nnDataset = generateDataset("regression", 80);
    
    weightLayer1Title.textContent = "W1 Weights Matrix (Hidden: 6 × Input: 1)";
    weightLayer2Title.textContent = "W2 Weights Matrix (Output: 1 × Hidden: 6)";
    nnPlotTitle.textContent = "Sine Wave Fit (Regression Curve)";
    nnPlotCaption.textContent = "Solid orange line = Neural Net prediction; Sky-blue dots = Target data points";
  } else {
    // 2D inputs for circular coordinates classification
    nnNet = new SimpleNeuralNet(2, 6, 1, "classification");
    nnDataset = generateDataset("classification", 100);

    weightLayer1Title.textContent = "W1 Weights Matrix (Hidden: 6 × Input: 2)";
    weightLayer2Title.textContent = "W2 Weights Matrix (Output: 1 × Hidden: 6)";
    nnPlotTitle.textContent = "Decision Boundary Space";
    nnPlotCaption.textContent = "Colored grid background shows classification boundary probability; Dots are target coordinates";
  }

  updateNNStats();
  updateNNPlots();
  renderNNParameters();
}

function trainSteps(steps) {
  let finalLoss = 0;
  for (let i = 0; i < steps; i++) {
    // Standard mini-batch update: train on the entire toy dataset as a single batch
    finalLoss = nnNet.trainStep(nnDataset, nnLr);
    nnEpoch++;
    nnLossHistory.push(finalLoss);
  }
  
  // Cap loss history array length for performance stability
  if (nnLossHistory.length > 500) {
    nnLossHistory.shift();
  }

  updateNNStats();
  updateNNPlots();
  renderNNParameters();
}

function toggleNNAutoplay() {
  if (nnAutoplayId === null) {
    btnTrainAuto.textContent = "Pause Training";
    btnTrainAuto.classList.add("active");
    btnTrainAuto.style.background = "var(--color-primary)";
    btnTrainAuto.style.color = "#ffffff";
    
    // Start loop
    const loop = () => {
      trainSteps(1); // 1 step per frame
      nnAutoplayId = requestAnimationFrame(loop);
    };
    nnAutoplayId = requestAnimationFrame(loop);
  } else {
    btnTrainAuto.textContent = "Auto Train (Play)";
    btnTrainAuto.classList.remove("active");
    btnTrainAuto.style.background = "rgba(139, 92, 246, 0.15)";
    btnTrainAuto.style.color = "#c084fc";
    
    cancelAnimationFrame(nnAutoplayId);
    nnAutoplayId = null;
  }
}

function updateNNStats() {
  statEpoch.textContent = nnEpoch;
  const currentLoss = nnLossHistory.length > 0 ? nnLossHistory[nnLossHistory.length - 1] : 0;
  statLoss.textContent = currentLoss.toFixed(5);
}

function updateNNPlots() {
  const mode = nnTaskSelect.value;
  drawFunctionFit(nnPlotCanvas, nnDataset, nnNet, mode);
  drawLossCurve(nnLossCanvas, nnLossHistory);
}

function renderNNParameters() {
  // Layer 1 weights grid: inputDim * hiddenDim
  weightLayer1Grid.innerHTML = "";
  nnNet.W1.forEach((row, i) => {
    row.forEach((val, j) => {
      const node = document.createElement("div");
      node.className = "weight-node";
      node.textContent = val.toFixed(2);
      
      // Dynamic shading representing values
      if (val > 0) {
        node.style.borderColor = `rgba(20, 184, 166, ${Math.min(0.8, val)})`;
        node.style.color = "#14b8a6";
      } else {
        node.style.borderColor = `rgba(239, 68, 68, ${Math.min(0.8, Math.abs(val))})`;
        node.style.color = "#f87171";
      }
      weightLayer1Grid.appendChild(node);
    });
  });

  // Layer 2 weights grid: hiddenDim * outputDim
  weightLayer2Grid.innerHTML = "";
  nnNet.W2.forEach((row, i) => {
    row.forEach((val, j) => {
      const node = document.createElement("div");
      node.className = "weight-node";
      node.textContent = val.toFixed(2);
      
      if (val > 0) {
        node.style.borderColor = `rgba(20, 184, 166, ${Math.min(0.8, val)})`;
        node.style.color = "#14b8a6";
      } else {
        node.style.borderColor = `rgba(239, 68, 68, ${Math.min(0.8, Math.abs(val))})`;
        node.style.color = "#f87171";
      }
      weightLayer2Grid.appendChild(node);
    });
  });
}

// Start application
window.addEventListener("DOMContentLoaded", init);
