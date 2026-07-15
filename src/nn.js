// Interactive PyTorch Concepts Demo - Neural Network Simulation
// Simulated in JS for learning purposes

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function tanh(x) {
  return Math.tanh(x);
}

export class SimpleNeuralNet {
  constructor(inputDim, hiddenDim, outputDim, taskMode = "regression") {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.outputDim = outputDim;
    this.taskMode = taskMode; // "regression" (sin fit) or "classification" (2-class circular blobs)
    this.reset();
  }

  reset() {
    // Initialize weights randomly using standard uniform/normal scaling (Glorot-like)
    this.W1 = [];
    for (let i = 0; i < this.hiddenDim; i++) {
      const row = [];
      for (let j = 0; j < this.inputDim; j++) {
        // Uniform in [-scale, scale]
        const scale = Math.sqrt(6 / (this.inputDim + this.hiddenDim));
        row.push((Math.random() * 2 - 1) * scale);
      }
      this.W1.push(row);
    }
    this.b1 = new Array(this.hiddenDim).fill(0);

    this.W2 = [];
    for (let i = 0; i < this.outputDim; i++) {
      const row = [];
      for (let j = 0; j < this.hiddenDim; j++) {
        const scale = Math.sqrt(6 / (this.hiddenDim + this.outputDim));
        row.push((Math.random() * 2 - 1) * scale);
      }
      this.W2.push(row);
    }
    this.b2 = new Array(this.outputDim).fill(0);
  }

  /**
   * Forward pass for a single input vector x.
   * Caches intermediate values for backpropagation.
   */
  forward(x) {
    const cache = { x: [...x] };

    // Layer 1: linear transformation
    const z1 = new Array(this.hiddenDim).fill(0);
    for (let i = 0; i < this.hiddenDim; i++) {
      let sum = 0;
      for (let j = 0; j < this.inputDim; j++) {
        sum += this.W1[i][j] * x[j];
      }
      z1[i] = sum + this.b1[i];
    }
    cache.z1 = z1;

    // Layer 1 activation (tanh)
    const a1 = z1.map(val => tanh(val));
    cache.a1 = a1;

    // Layer 2: linear transformation
    const z2 = new Array(this.outputDim).fill(0);
    for (let i = 0; i < this.outputDim; i++) {
      let sum = 0;
      for (let j = 0; j < this.hiddenDim; j++) {
        sum += this.W2[i][j] * a1[j];
      }
      z2[i] = sum + this.b2[i];
    }
    cache.z2 = z2;

    // Layer 2 activation (identity for regression, sigmoid for classification)
    let a2;
    if (this.taskMode === "classification") {
      a2 = z2.map(val => sigmoid(val));
    } else {
      a2 = [...z2]; // regression: identity activation
    }
    cache.a2 = a2;

    return { output: a2, cache: cache };
  }

  /**
   * Computes gradients for a single training example (x, y) given the cached forward values.
   * Loss function: MSE (Mean Squared Error).
   */
  backward(cache, target) {
    const grad = {
      dW1: this.W1.map(row => row.map(() => 0)),
      db1: new Array(this.hiddenDim).fill(0),
      dW2: this.W2.map(row => row.map(() => 0)),
      db2: new Array(this.outputDim).fill(0)
    };

    const x = cache.x;
    const a1 = cache.a1;
    const a2 = cache.a2;

    // Output loss gradients (MSE loss derivative w.r.t model predictions)
    // Loss = 0.5 * sum((pred_i - target_i)^2)
    // dLoss/dPred = pred - target
    const dLoss_da2 = new Array(this.outputDim).fill(0);
    for (let i = 0; i < this.outputDim; i++) {
      dLoss_da2[i] = a2[i] - target[i];
    }

    // Gradient w.r.t pre-activation z2
    const dLoss_dz2 = new Array(this.outputDim).fill(0);
    for (let i = 0; i < this.outputDim; i++) {
      if (this.taskMode === "classification") {
        // Derivative of Sigmoid is a2 * (1 - a2)
        dLoss_dz2[i] = dLoss_da2[i] * a2[i] * (1.0 - a2[i]);
      } else {
        // Regression: Identity derivative is 1
        dLoss_dz2[i] = dLoss_da2[i];
      }
    }

    // Gradients w.r.t W2 and b2
    // dLoss/dW2 = dLoss/dz2 * a1^T
    // dLoss/db2 = dLoss/dz2
    for (let i = 0; i < this.outputDim; i++) {
      for (let j = 0; j < this.hiddenDim; j++) {
        grad.dW2[i][j] = dLoss_dz2[i] * a1[j];
      }
      grad.db2[i] = dLoss_dz2[i];
    }

    // Gradient backpropagated to hidden layer activation a1
    // dLoss/da1 = W2^T * dLoss/dz2
    const dLoss_da1 = new Array(this.hiddenDim).fill(0);
    for (let j = 0; j < this.hiddenDim; j++) {
      let sum = 0;
      for (let i = 0; i < this.outputDim; i++) {
        sum += this.W2[i][j] * dLoss_dz2[i];
      }
      dLoss_da1[j] = sum;
    }

    // Gradient w.r.t pre-activation z1
    // dLoss/dz1 = dLoss/da1 * (1 - tanh^2(z1)) = dLoss/da1 * (1 - a1^2)
    const dLoss_dz1 = new Array(this.hiddenDim).fill(0);
    for (let j = 0; j < this.hiddenDim; j++) {
      dLoss_dz1[j] = dLoss_da1[j] * (1.0 - a1[j] * a1[j]);
    }

    // Gradients w.r.t W1 and b1
    // dLoss/dW1 = dLoss/dz1 * x^T
    // dLoss/db1 = dLoss/dz1
    for (let i = 0; i < this.hiddenDim; i++) {
      for (let j = 0; j < this.inputDim; j++) {
        grad.dW1[i][j] = dLoss_dz1[i] * x[j];
      }
      grad.db1[i] = dLoss_dz1[i];
    }

    return grad;
  }

  /**
   * Performs training step over a mini-batch of items.
   * Accumulates gradients across the batch and updates weights using standard gradient descent.
   */
  trainStep(batchData, learningRate) {
    const accumGrad = {
      dW1: this.W1.map(row => row.map(() => 0)),
      db1: new Array(this.hiddenDim).fill(0),
      dW2: this.W2.map(row => row.map(() => 0)),
      db2: new Array(this.outputDim).fill(0)
    };

    let totalLoss = 0;
    const batchSize = batchData.length;

    batchData.forEach(item => {
      const { output, cache } = this.forward(item.x);
      
      // Calculate Loss (MSE)
      let loss = 0;
      for (let i = 0; i < this.outputDim; i++) {
        const diff = output[i] - item.y[i];
        loss += 0.5 * diff * diff;
      }
      totalLoss += loss;

      // Compute gradients via manual backprop
      const stepGrad = this.backward(cache, item.y);

      // Accumulate gradients
      for (let i = 0; i < this.hiddenDim; i++) {
        for (let j = 0; j < this.inputDim; j++) {
          accumGrad.dW1[i][j] += stepGrad.dW1[i][j];
        }
        accumGrad.db1[i] += stepGrad.db1[i];
      }
      for (let i = 0; i < this.outputDim; i++) {
        for (let j = 0; j < this.hiddenDim; j++) {
          accumGrad.dW2[i][j] += stepGrad.dW2[i][j];
        }
        accumGrad.db2[i] += stepGrad.db2[i];
      }
    });

    // Update parameters with average gradients
    const normFactor = 1.0 / batchSize;
    for (let i = 0; i < this.hiddenDim; i++) {
      for (let j = 0; j < this.inputDim; j++) {
        this.W1[i][j] -= learningRate * accumGrad.dW1[i][j] * normFactor;
      }
      this.b1[i] -= learningRate * accumGrad.db1[i] * normFactor;
    }
    for (let i = 0; i < this.outputDim; i++) {
      for (let j = 0; j < this.hiddenDim; j++) {
        this.W2[i][j] -= learningRate * accumGrad.dW2[i][j] * normFactor;
      }
      this.b2[i] -= learningRate * accumGrad.db2[i] * normFactor;
    }

    return totalLoss / batchSize; // Average loss
  }
}

/**
 * Generates synthetic datasets for training.
 */
export function generateDataset(taskMode, size = 100) {
  const dataset = [];

  if (taskMode === "regression") {
    // y = sin(x) + gaussian noise
    for (let i = 0; i < size; i++) {
      // x values between -3.14 and 3.14
      const xVal = -Math.PI + (2 * Math.PI * i) / (size - 1);
      const cleanY = Math.sin(xVal);
      // Small Gaussian noise
      const noise = (Math.random() * 2 - 1) * 0.15;
      dataset.push({
        x: [xVal],
        y: [cleanY + noise]
      });
    }
  } else {
    // Classification: Concentric circles (Circle dataset)
    // Class 0: inner ring/circle (radius < 0.5)
    // Class 1: outer ring (radius 0.65 to 1.0)
    for (let i = 0; i < size; i++) {
      const label = Math.random() < 0.5 ? 0 : 1;
      let r, theta;
      if (label === 0) {
        // Inner circle
        r = Math.random() * 0.45;
      } else {
        // Outer ring
        r = 0.6 + Math.random() * 0.35;
      }
      theta = Math.random() * 2 * Math.PI;

      // Map to coordinates (with a small random noise)
      const x1 = r * Math.cos(theta) + (Math.random() - 0.5) * 0.05;
      const x2 = r * Math.sin(theta) + (Math.random() - 0.5) * 0.05;

      dataset.push({
        x: [x1, x2],
        y: [label]
      });
    }
  }

  return dataset;
}
