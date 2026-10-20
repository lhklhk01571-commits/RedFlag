import { COLS, COLORS, ROWS, SHAPES } from "./engine.js";

const GAP = 1;

export function drawBoard(ctx, state, width, height) {
  const cell = width / COLS;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#10151e";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(232, 237, 245, 0.05)";
  ctx.lineWidth = 1;
  for (let x = 1; x < COLS; x += 1) {
    ctx.beginPath();
    ctx.moveTo(x * cell + 0.5, 0);
    ctx.lineTo(x * cell + 0.5, height);
    ctx.stroke();
  }
  for (let y = 1; y < ROWS; y += 1) {
    ctx.beginPath();
    ctx.moveTo(0, y * cell + 0.5);
    ctx.lineTo(width, y * cell + 0.5);
    ctx.stroke();
  }

  for (let y = 0; y < ROWS; y += 1) {
    for (let x = 0; x < COLS; x += 1) {
      const type = state.board[y][x];
      if (type) drawBlock(ctx, x, y, cell, COLORS[type], 1);
    }
  }

  const active = new Set(state.active.map((cellPos) => `${cellPos.x},${cellPos.y}`));
  for (const cellPos of state.ghost) {
    if (cellPos.y < 0 || active.has(`${cellPos.x},${cellPos.y}`)) continue;
    drawBlock(ctx, cellPos.x, cellPos.y, cell, COLORS[state.piece.type], 0.22);
  }
  for (const cellPos of state.active) {
    if (cellPos.y < 0) continue;
    drawBlock(ctx, cellPos.x, cellPos.y, cell, COLORS[cellPos.type || state.piece.type], 1);
  }
}

export function drawPiece(ctx, type, width, height) {
  ctx.clearRect(0, 0, width, height);
  if (!type) return;
  const matrix = SHAPES[type];
  const filled = [];
  for (let y = 0; y < matrix.length; y += 1) {
    for (let x = 0; x < matrix[y].length; x += 1) {
      if (matrix[y][x]) filled.push({ x, y });
    }
  }
  const minX = Math.min(...filled.map((cell) => cell.x));
  const maxX = Math.max(...filled.map((cell) => cell.x));
  const minY = Math.min(...filled.map((cell) => cell.y));
  const maxY = Math.max(...filled.map((cell) => cell.y));
  const cols = maxX - minX + 1;
  const rows = maxY - minY + 1;
  const cell = Math.min(width / (cols + 0.6), height / (rows + 0.6), 28);
  const originX = (width - cols * cell) / 2 - minX * cell;
  const originY = (height - rows * cell) / 2 - minY * cell;
  for (const block of filled) {
    drawBlock(ctx, block.x, block.y, cell, COLORS[type], 1, originX, originY);
  }
}

function drawBlock(ctx, x, y, cell, color, alpha, originX = 0, originY = 0) {
  const px = originX + x * cell + GAP;
  const py = originY + y * cell + GAP;
  const size = cell - GAP * 2;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(px, py, size, size, 4);
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.28)";
  ctx.fillRect(px, py, size, Math.max(3, size * 0.16));
  ctx.fillStyle = "rgba(0,0,0,0.16)";
  ctx.fillRect(px, py + size * 0.82, size, Math.max(3, size * 0.18));
  ctx.restore();
}
