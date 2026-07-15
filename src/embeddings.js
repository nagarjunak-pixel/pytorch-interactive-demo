// Interactive PyTorch Concepts Demo - Embedding & Similarity Logic
// Simulated in JS for learning purposes

export const SEMANTIC_DIMENSIONS = [
  "AI & Machine Learning",   // Dim 0
  "PyTorch & Libraries",    // Dim 1
  "NLP & Human Text",       // Dim 2
  "Vector DB & Retrieval",  // Dim 3
  "Math & Similarity",      // Dim 4
  "Neural Networks",        // Dim 5
  "Data & Training",        // Dim 6
  "Software & Systems"      // Dim 7
];

const DIMENSION_KEYWORDS = [
  // AI & ML
  ["machine", "learning", "artificial", "intelligence", "ai", "ml", "subset", "broader", "field"],
  // PyTorch & Libraries
  ["pytorch", "library", "torch", "code", "programming", "open-source", "framework"],
  // NLP & Human Text
  ["natural", "language", "processing", "nlp", "text", "human", "word", "sentences", "speech"],
  // Vector DB & Retrieval
  ["vector", "database", "databases", "rag", "retrieval", "store", "pinecone", "weaviate", "pgvector", "retrieve", "index"],
  // Math & Similarity
  ["similarity", "cosine", "angle", "distance", "dot", "measures", "metric", "geometry"],
  // Neural Networks
  ["neural", "network", "networks", "brain", "inspired", "neurons", "layers", "weights", "biases"],
  // Data & Training
  ["data", "dataset", "train", "training", "fit", "loss", "optimizer", "epochs", "backprop"],
  // Software & Systems
  ["software", "system", "pipeline", "pipelines", "deploy", "server", "cloud", "production"]
];

/**
 * Deterministically generates an 8-dimensional embedding vector for a given text.
 * Uses keyword matching and fallback character hashing to ensure a dense vector.
 */
export function embed(text) {
  const vec = new Array(8).fill(0);
  if (!text) return vec;

  const cleanText = text.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, "");
  const words = cleanText.split(/\s+/).filter(w => w.length > 0);

  if (words.length === 0) return vec;

  // 1. Keyword-based matching (semantic projection)
  words.forEach(word => {
    DIMENSION_KEYWORDS.forEach((keywords, dimIndex) => {
      if (keywords.includes(word)) {
        vec[dimIndex] += 1.5;
      }
      // Substring match for plural or root words
      else {
        keywords.forEach(keyword => {
          if (word.length > 3 && keyword.length > 3 && (word.includes(keyword) || keyword.includes(word))) {
            vec[dimIndex] += 0.5;
          }
        });
      }
    });
  });

  // 2. Character-hash fallback to avoid sparse/zero vectors and add "uniqueness"
  for (let i = 0; i < cleanText.length; i++) {
    const charCode = cleanText.charCodeAt(i);
    const dim = charCode % 8;
    vec[dim] += 0.05;
  }

  // 3. Normalize the vector to unit length (L2 norm)
  let sumSq = 0;
  for (let i = 0; i < 8; i++) {
    sumSq += vec[i] * vec[i];
  }
  const norm = Math.sqrt(sumSq);

  if (norm > 0) {
    for (let i = 0; i < 8; i++) {
      vec[i] = vec[i] / norm;
    }
  }

  return vec;
}

/**
 * Computes the cosine similarity between two vectors.
 * Since they are pre-normalized, this is equivalent to the dot product.
 */
export function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  for (let i = 0; i < 8; i++) {
    dotProduct += vecA[i] * vecB[i];
  }
  return dotProduct; // Returns value in range [-1, 1]
}

/**
 * Projects an 8D vector onto a 2D plane for visualization.
 * Maps each dimension to an angle theta_i = i * 2 * pi / 8.
 * Returns coordinates {x, y} mapped roughly to [-1, 1].
 */
export function project2D(vector) {
  let x = 0;
  let y = 0;

  for (let i = 0; i < 8; i++) {
    const theta = (i * 2 * Math.PI) / 8;
    const w = vector[i];
    x += w * Math.cos(theta);
    y += w * Math.sin(theta);
  }

  // Scale the output to keep it visually spread out
  return { x: x, y: y };
}
