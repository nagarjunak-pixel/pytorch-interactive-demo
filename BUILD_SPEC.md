# BUILD SPEC — Interactive PyTorch Concepts Demo (Gemini/agy build task)

You are building a SELF-CONTAINED, RUNTIME-ONLY (no server, no build step, no npm,
no Python backend) interactive educational web demo that teaches two PyTorch
concepts the user just learned in a hands-on, simulated way. The user is a
beginner who learned PyTorch step-by-step (tensors → autograd → nn.Module).
Everything must run in the browser by opening index.html (or after a Netlify
static deploy). NO external network calls. NO CDN dependencies that would fail
offline — inline or self-host ALL JS/CSS. Use vanilla JS + Canvas 2D only.

## Hard constraints
- Single static site: `index.html`, `style.css`, `app.js`, and a `src/` folder of
  plain JS modules loaded via `<script type="module">`. No bundler, no framework
  (no React/Vue), no npm install. It MUST open with `file://` and also work on
  Netlify static hosting.
- NO PyTorch / no Python at runtime. The "neural network" and "embeddings" are
  SIMULATED in JavaScript so the page is interactive offline. Be explicit in the
  UI that these are *simulated in JS to teach the concept* (the user explicitly
  asked for a "simulated and interactive demo to understand the concepts").
- Plain vanilla JS + HTML + CSS + Canvas. Keep it readable; this is a teaching tool.
- Mobile-friendly enough to view on a phone (responsive, max-width container).

## Concept A — Embeddings & Similarity (maps to FDE "RAG pipelines / embedding models")
Build an interactive panel that teaches: text/items get turned into vectors
(embeddings); similarity = distance between vectors; retrieval = find nearest.
Interactive elements:
1. A small editable list of ~6-8 example "documents" (short sentences/words).
2. A JS function `embed(text)` that deterministically turns text into a fixed
   vector (e.g., bag-of-chars or hashed bag-of-words into a 8–16 dim vector,
   normalized). Show the actual vector numbers in the UI.
3. A query box. On query, compute cosine similarity between query embedding and
   each doc embedding, display a ranked list (top matches first) with similarity
   bars. This is literally how RAG retrieval works — label it as such.
4. A 2D visualization (Canvas): plot each doc as a point using its first 2
   embedding dims; plot the query as a distinct marker; draw lines to the top-3
   nearest. Show that "nearby points = similar meaning."
5. Include a SHORT plain-language explainer (3-4 bullets) tying it to the FDE
   slide: "RAG pipelines — chunking, vector databases (Pinecone/Weaviate/pgvector),
   embedding models, reranking."

## Concept B — Neural Network training (maps to the nn.Module lesson, sin fit)
Build an interactive panel that teaches the training loop:
FORWARD → LOSS → BACKWARD → STEP, with a live loss curve.
Interactive elements:
1. A tiny 2-layer net simulated in JS: input→hidden(linear+tanh/relu)→output.
   Implement forward pass, MSE loss, and a manual backprop (chain rule) OR a
   simple numerical/analytic gradient descent over the few weights. Keep weights
   small (hidden size ~6) so backprop is simple and correct.
2. Data: generate y = sin(x) + noise (same toy as the user's lesson 3) OR a
   simple 2-class spiral for classification. Default to sin(x) to match their
   lesson, but offer a toggle to "2-class classification" mode that shows a
   decision boundary (this connects to the MNIST idea they wanted too).
3. Controls: LEARN RATE slider, BATCH/STEP button, "Train 1 step", "Train 50
   steps", RESET. Show the current weights/biases as numbers.
4. Live loss curve (Canvas line chart) updating each step — user watches loss fall.
5. Plot: true curve (dashed) vs current network prediction (solid red),
   updating live. This is the EXACT picture from their lesson-3 output
   (03_fit.png) — reproduce that look.
6. SHORT explainer bullets: "FORWARD computes prediction; BACKWARD computes
   gradients (∂loss/∂w); STEP nudges weights downhill (w -= lr·grad). This 3-move
   loop is under every fine-tune and embedding model."

## Shared page structure
- Header: "PyTorch Concepts — Interactive Demo" + one line: "Simulated in
  JavaScript to teach the ideas (no Python running here)."
- A top nav / tab switch: [A: Embeddings] [B: Neural Net] and an [Intro/How it
  works] tab with the 3-layer PyTorch diagram (tensor→autograd→nn.Module) as an
  HTML/CSS diagram.
- Each panel has: the interactive area, a live explainer, and a "what you just
  saw" recap.
- A footer linking back to the local lessons (mention ~/pytorch_learn/lessons).
- Keep copy beginner-friendly and in the same teaching voice as a step-by-step
  guide. Use simple diagrams (CSS boxes/arrows) in the Intro tab.

## Acceptance / self-check (agy must verify before finishing)
1. All files present: index.html, style.css, app.js, src/*.js.
2. `node --check` passes on every .js file (no syntax errors).
3. Open index.html headless (you may use a quick check by loading via node with
   jsdom OR simply ensure no top-level runtime errors by reasoning + node --check;
   if you have a headless browser available, screenshot both tabs).
4. The embeddings panel actually ranks docs by cosine similarity (verify the
   math in code).
5. The NN panel actually reduces loss over steps (verify the loop is correct).
6. Print a final directory tree and a one-line "self-check: PASS/FAIL" summary.

Output everything into: /Users/venkataswaraswamy/pytorch_learn/interactive_demo/
Do NOT push to git or deploy — Hermes handles GitHub + Netlify after your build.
