// Interactive PyTorch Concepts Demo - Canvas Visualizations
// Simulated in JS for learning purposes

import { project2D, SEMANTIC_DIMENSIONS } from "./embeddings.js";

/**
 * Renders the 2D projected embedding space.
 */
export function drawEmbeddingSpace(canvas, docs, queryDoc, top3Indexes) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const radius = Math.min(w, h) * 0.38; // Radius scale

  // 1. Clear background
  ctx.fillStyle = "#0f172a"; // slate-900
  ctx.fillRect(0, 0, w, h);

  // 2. Draw circular grid lines and radar directions
  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;
  
  // Draw concentric rings
  for (let rFactor of [0.33, 0.66, 1.0]) {
    ctx.beginPath();
    ctx.arc(cx, cy, radius * rFactor, 0, 2 * Math.PI);
    ctx.stroke();
  }

  // Draw 8 axes and labels
  SEMANTIC_DIMENSIONS.forEach((dimName, i) => {
    const theta = (i * 2 * Math.PI) / 8;
    const ax = cx + radius * Math.cos(theta);
    const ay = cy - radius * Math.sin(theta); // Subtract because Canvas y increases downwards

    // Axis line
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(ax, ay);
    ctx.stroke();

    // Axis label
    ctx.fillStyle = "rgba(148, 163, 184, 0.5)"; // slate-400 with opacity
    ctx.font = "8px sans-serif";
    const labelDistance = radius + 18;
    const lx = cx + labelDistance * Math.cos(theta);
    const ly = cy - labelDistance * Math.sin(theta);
    
    ctx.textAlign = Math.abs(Math.cos(theta)) < 0.1 ? "center" : Math.cos(theta) > 0 ? "left" : "right";
    ctx.textBaseline = "middle";
    
    // Shorten label for layout neatness
    let displayLabel = dimName.split(" & ")[0];
    ctx.fillText(displayLabel, lx, ly);
  });

  const queryCoord = queryDoc ? project2D(queryDoc.vector) : null;
  const queryCx = queryCoord ? cx + queryCoord.x * radius : null;
  const queryCy = queryCoord ? cy - queryCoord.y * radius : null;

  // 3. Draw lines from Query to Top 3 Matches (if query exists)
  if (queryCx !== null && queryCy !== null && top3Indexes.length > 0) {
    top3Indexes.forEach((idx, order) => {
      const doc = docs[idx];
      if (!doc) return;
      const coord = project2D(doc.vector);
      const docCx = cx + coord.x * radius;
      const docCy = cy - coord.y * radius;

      // Draw dashed connection line
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.moveTo(queryCx, queryCy);
      ctx.lineTo(docCx, docCy);
      
      // Color intensity based on order (top 1 is brighter)
      if (order === 0) {
        ctx.strokeStyle = "rgba(255, 102, 51, 0.7)"; // PyTorch orange
        ctx.lineWidth = 2;
      } else {
        ctx.strokeStyle = "rgba(139, 92, 246, 0.5)"; // Violet
        ctx.lineWidth = 1;
      }
      ctx.stroke();
    });
    ctx.setLineDash([]); // Reset line dash
  }

  // 4. Plot Documents
  docs.forEach((doc, idx) => {
    if (!doc.vector) return;
    const coord = project2D(doc.vector);
    const docCx = cx + coord.x * radius;
    const docCy = cy - coord.y * radius;

    const isTopMatch = top3Indexes.includes(idx);
    const isTop1 = top3Indexes[0] === idx;

    // Draw node bubble
    ctx.beginPath();
    ctx.arc(docCx, docCy, isTopMatch ? 9 : 7, 0, 2 * Math.PI);
    
    if (isTop1) {
      ctx.fillStyle = "#ff6633"; // PyTorch orange
    } else if (isTopMatch) {
      ctx.fillStyle = "#8b5cf6"; // Purple
    } else {
      ctx.fillStyle = "rgba(30, 41, 59, 0.85)"; // dark slate
    }
    ctx.fill();

    ctx.strokeStyle = isTopMatch ? "#ffffff" : "rgba(255, 255, 255, 0.3)";
    ctx.lineWidth = isTopMatch ? 1.5 : 1;
    ctx.stroke();

    // Node Index Text
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 9px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText((idx + 1).toString(), docCx, docCy);
  });

  // 5. Plot Query Document
  if (queryCx !== null && queryCy !== null) {
    ctx.fillStyle = "#ffdd57"; // query color (gold/yellow)
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;

    // Draw query as a star or distinctive diamond
    ctx.beginPath();
    const size = 8;
    ctx.moveTo(queryCx, queryCy - size);
    ctx.lineTo(queryCx + size, queryCy);
    ctx.lineTo(queryCx, queryCy + size);
    ctx.lineTo(queryCx - size, queryCy);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Query Label
    ctx.fillStyle = "#ffdd57";
    ctx.font = "bold 10px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText("Query", queryCx, queryCy - 11);
  }
}

/**
 * Draws the NN prediction curve (for sine wave) or decision boundary (for circles).
 */
export function drawFunctionFit(canvas, dataset, net, mode) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  // 1. Clear background
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, w, h);

  if (mode === "regression") {
    // Coordinate mapping functions (x: [-Math.PI, Math.PI] -> [30, w-20], y: [-1.4, 1.4] -> [h-20, 20])
    const padXLeft = 35;
    const padXRight = 20;
    const padY = 25;

    const toCanvasX = (xVal) => {
      const pct = (xVal - (-Math.PI)) / (2 * Math.PI);
      return padXLeft + pct * (w - padXLeft - padXRight);
    };

    const toCanvasY = (yVal) => {
      const pct = (yVal - (-1.4)) / 2.8; // Range size 2.8
      return h - padY - pct * (h - 2 * padY);
    };

    // Draw axes
    ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
    ctx.lineWidth = 1;
    
    // Y=0 Axis
    ctx.beginPath();
    ctx.moveTo(padXLeft, toCanvasY(0));
    ctx.lineTo(w - padXRight, toCanvasY(0));
    ctx.stroke();

    // X=0 Axis
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), padY);
    ctx.lineTo(toCanvasX(0), h - padY);
    ctx.stroke();

    // Draw axis ticks and text labels
    ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
    ctx.font = "9px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("-π", toCanvasX(-Math.PI), toCanvasY(0) + 4);
    ctx.fillText("0", toCanvasX(0) + 4, toCanvasY(0) + 4);
    ctx.fillText("π", toCanvasX(Math.PI), toCanvasY(0) + 4);

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText("1.0", toCanvasX(-Math.PI) - 5, toCanvasY(1.0));
    ctx.fillText("-1.0", toCanvasX(-Math.PI) - 5, toCanvasY(-1.0));

    // A. Draw True Curve (Dashed Sine Wave)
    ctx.beginPath();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.35)"; // dashed slate
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 5]);
    
    const steps = 100;
    for (let i = 0; i <= steps; i++) {
      const xVal = -Math.PI + (2 * Math.PI * i) / steps;
      const yVal = Math.sin(xVal);
      const cx = toCanvasX(xVal);
      const cy = toCanvasY(yVal);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
    ctx.setLineDash([]); // reset

    // B. Draw Training Points
    dataset.forEach(item => {
      const cx = toCanvasX(item.x[0]);
      const cy = toCanvasY(item.y[0]);
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(56, 189, 248, 0.55)"; // sky-400 dot
      ctx.fill();
    });

    // C. Draw Neural Network Prediction Curve (Solid Red/Orange)
    ctx.beginPath();
    ctx.strokeStyle = "#ff6633"; // PyTorch Orange
    ctx.lineWidth = 3;
    
    for (let i = 0; i <= steps; i++) {
      const xVal = -Math.PI + (2 * Math.PI * i) / steps;
      const { output } = net.forward([xVal]);
      const cx = toCanvasX(xVal);
      const cy = toCanvasY(output[0]);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();

  } else {
    // classification: 2D circular boundary
    // Mapping: grid of coordinates [-1.2, 1.2]
    const minCoord = -1.25;
    const maxCoord = 1.25;
    const range = maxCoord - minCoord;

    const toCanvasX = (x1) => 30 + ((x1 - minCoord) / range) * (w - 50);
    const toCanvasY = (x2) => h - 30 - ((x2 - minCoord) / range) * (h - 50);

    // A. Draw background decision boundary
    const gridSize = 45;
    const cellW = (w - 50) / gridSize;
    const cellH = (h - 50) / gridSize;

    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        // Find coordinate at center of this cell
        const gridX = minCoord + ((i + 0.5) / gridSize) * range;
        const gridY = minCoord + ((j + 0.5) / gridSize) * range;

        // Neural network forward pass
        const { output } = net.forward([gridX, gridY]);
        const p = output[0]; // Sigmoid output [0, 1]

        // Color cell: blend red (class 0) and blue (class 1)
        // Red component: high when p is low. Blue component: high when p is high.
        // We use low opacity so the grid points show clearly
        const rColor = Math.round((1 - p) * 110 + 20);
        const bColor = Math.round(p * 110 + 20);
        const gColor = 20;

        ctx.fillStyle = `rgb(${rColor}, ${gColor}, ${bColor})`;
        
        // Draw the cell
        const cx = 30 + i * cellW;
        const cy = h - 30 - (j + 1) * cellH; // flip Y
        ctx.fillRect(cx, cy, cellW + 0.5, cellH + 0.5);
      }
    }

    // Draw central circular rings as helper guides
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(toCanvasX(0), toCanvasY(0), ((0.5) / range) * (w - 50), 0, 2 * Math.PI);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(toCanvasX(0), toCanvasY(0), ((0.8) / range) * (w - 50), 0, 2 * Math.PI);
    ctx.stroke();

    // B. Draw Training Points
    dataset.forEach(item => {
      const cx = toCanvasX(item.x[0]);
      const cy = toCanvasY(item.x[1]);
      const label = item.y[0];

      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
      
      if (label === 0) {
        ctx.fillStyle = "#ef4444"; // red-500
        ctx.strokeStyle = "#fee2e2";
      } else {
        ctx.fillStyle = "#3b82f6"; // blue-500
        ctx.strokeStyle = "#dbeafe";
      }
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Draw axis descriptions
    ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
    ctx.font = "8px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("x1", w - 18, toCanvasY(0) + 3);
    ctx.textAlign = "center";
    ctx.fillText("x2", toCanvasX(0), 10);
  }
}

/**
 * Draws the Loss History Line Chart.
 */
export function drawLossCurve(canvas, lossHistory) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const padLeft = 40;
  const padRight = 15;
  const padBottom = 20;
  const padTop = 15;

  // 1. Clear background
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, w, h);

  // 2. Base coordinate plane
  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padLeft, padTop);
  ctx.lineTo(padLeft, h - padBottom);
  ctx.lineTo(w - padRight, h - padBottom);
  ctx.stroke();

  if (lossHistory.length === 0) {
    ctx.fillStyle = "rgba(148, 163, 184, 0.4)";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Waiting for training...", w / 2, h / 2);
    return;
  }

  // 3. Compute limits
  const maxLoss = Math.max(0.3, ...lossHistory) * 1.05; // Force at least 0.3 max scale
  const minLoss = 0;
  const length = lossHistory.length;

  const toCanvasX = (idx) => {
    if (length <= 1) return padLeft;
    return padLeft + (idx / (length - 1)) * (w - padLeft - padRight);
  };

  const toCanvasY = (lossVal) => {
    const pct = lossVal / maxLoss;
    return h - padBottom - pct * (h - padBottom - padTop);
  };

  // Draw grid helper lines (horizontal)
  ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
  ctx.lineWidth = 1;
  for (let val of [maxLoss * 0.25, maxLoss * 0.5, maxLoss * 0.75, maxLoss]) {
    ctx.beginPath();
    ctx.moveTo(padLeft, toCanvasY(val));
    ctx.lineTo(w - padRight, toCanvasY(val));
    ctx.stroke();
  }

  // Label axes
  ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText("0.0", padLeft - 6, toCanvasY(0));
  ctx.fillText((maxLoss * 0.5).toFixed(2), padLeft - 6, toCanvasY(maxLoss * 0.5));
  ctx.fillText(maxLoss.toFixed(2), padLeft - 6, toCanvasY(maxLoss));

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("0", padLeft, h - padBottom + 4);
  ctx.fillText(length.toString(), w - padRight, h - padBottom + 4);

  // 4. Draw loss history line
  ctx.beginPath();
  ctx.strokeStyle = "#a78bfa"; // violet-400
  ctx.lineWidth = 2;

  lossHistory.forEach((loss, i) => {
    const cx = toCanvasX(i);
    const cy = toCanvasY(loss);
    if (i === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  });
  ctx.stroke();

  // Highlight current loss text inside graph
  const currentLoss = lossHistory[length - 1];
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 9px sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "bottom";
  ctx.fillText(`Loss: ${currentLoss.toFixed(5)}`, w - padRight - 5, padTop + 8);
}
