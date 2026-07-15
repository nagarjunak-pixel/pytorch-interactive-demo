# PyTorch Concepts — Interactive Demo

A hands-on, **simulated** (in-browser JavaScript, no Python/server) interactive demo
to understand the core PyTorch ideas behind Field Deployment Engineering:

- **Tab 1 — PyTorch Trilogy**: tensor to autograd to nn.Module mental model.
- **Tab 2 — Embeddings & RAG**: type a query, watch cosine-similarity retrieval rank
  documents (the core of RAG pipelines / vector DBs on the FDE skill list).
- **Tab 3 — NN Training Loop**: train a tiny 2-layer net live — watch the loss curve
  fall and the prediction fit sin(x) (or a 2-class decision boundary).

## Run locally
Open `index.html` directly, or serve it:
```
python3 -m http.server 8000
# visit http://localhost:8000
```
No build step, no dependencies, no network calls.

## Companion lessons (real PyTorch, on your machine)
`~/pytorch_learn/lessons/01_tensors.py` through `03_nn_module.py` — run with
`source ~/pytorch_learn/venv/bin/activate && python lessons/03_nn_module.py`
