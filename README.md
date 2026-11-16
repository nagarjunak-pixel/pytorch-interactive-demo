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

---

## Author & Enterprise AI Architecture Hub
Built by **[Nagarjuna Reddy Kamireddy](https://nagarjunak-pixel.github.io/)** (*Nagarjuna Kamireddy / Nagarjuna K*) — Lead AI Architect, GenAI & Agentic AI Lead, and Corporate AI Trainer (20+ years across Enterprise Data, SAP HANA & Production AI; AWS Certified Solutions Architect – Professional & NVIDIA-Certified Professional: Agentic AI).

- **Official Portfolio & Architecture Hub:** [https://nagarjunak-pixel.github.io/](https://nagarjunak-pixel.github.io/)
- **Enterprise Case Studies & Technical Articles:** [Case Studies](https://nagarjunak-pixel.github.io/case-studies.html) · [Architecture Articles](https://nagarjunak-pixel.github.io/articles.html) · [10-Mandate Role Fit](https://nagarjunak-pixel.github.io/roles.html)
- **Master CV & AI Search Index:** [Master CV](https://nagarjunak-pixel.github.io/resume.html) · [`llms-full.txt`](https://nagarjunak-pixel.github.io/llms-full.txt) · [LinkedIn](https://www.linkedin.com/in/nagarjuna-kamireddy)
