export const COLS = 10;
export const ROWS = 20;

export const SHAPES = {
  I: [
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ],
  O: [
    [1, 1],
    [1, 1],
  ],
  T: [
    [0, 1, 0],
    [1, 1, 1],
    [0, 0, 0],
  ],
  S: [
    [0, 1, 1],
    [1, 1, 0],
    [0, 0, 0],
  ],
  Z: [
    [1, 1, 0],
    [0, 1, 1],
    [0, 0, 0],
  ],
  J: [
    [1, 0, 0],
    [1, 1, 1],
    [0, 0, 0],
  ],
  L: [
    [0, 0, 1],
    [1, 1, 1],
    [0, 0, 0],
  ],
};

export const COLORS = {
  I: "#2ec4d6",
  O: "#f0c14e",
  T: "#c46bdc",
  S: "#5dbe6e",
  Z: "#e85d6c",
  J: "#4c7fe8",
  L: "#ef8a3a",
};

const PIECES = Object.keys(SHAPES);
const SPAWN_X = { I: 3, O: 4, T: 3, S: 3, Z: 3, J: 3, L: 3 };
const LINE_POINTS = [0, 100, 300, 500, 800];

export function dropInterval(level) {
  return Math.max(100, 800 - (level - 1) * 70);
}

export function lineScore(cleared, level) {
  return (LINE_POINTS[cleared] ?? 0) * level;
}

export function rotateMatrix(matrix, direction) {
  const turns = direction < 0 ? 3 : 1;
  let next = matrix;
  for (let i = 0; i < turns; i += 1) next = rotateCW(next);
  return next;
}

function rotateCW(matrix) {
  const n = matrix.length;
  const next = Array.from({ length: n }, () => Array(n).fill(0));
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) next[x][n - 1 - y] = matrix[y][x];
  }
  return next;
}

function emptyBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(null));
}

function cloneMatrix(matrix) {
  return matrix.map((row) => row.slice());
}

export function createGame(options = {}) {
  const random = options.random ?? Math.random;
  const source = options.sequence ? options.sequence.slice() : null;

  let board = emptyBoard();
  let sequence = [];
  let bag = [];
  let queue = [];
  let piece = null;
  let holdType = null;
  let holdUsed = false;
  let score = 0;
  let lines = 0;
  let level = 1;
  let status = "ready";
  let accumulator = 0;
  let lastClear = 0;
  let lastPoints = 0;

  function reset() {
    board = emptyBoard();
    sequence = source ? source.slice() : [];
    bag = [];
    queue = [];
    piece = null;
    holdType = null;
    holdUsed = false;
    score = 0;
    lines = 0;
    level = 1;
    status = "ready";
    accumulator = 0;
    lastClear = 0;
    lastPoints = 0;
  }

  function pull() {
    if (sequence.length) return sequence.shift();
    if (!bag.length) {
      const next = PIECES.slice();
      for (let i = next.length - 1; i > 0; i -= 1) {
        const j = Math.floor(random() * (i + 1));
        [next[i], next[j]] = [next[j], next[i]];
      }
      bag = next;
    }
    return bag.shift();
  }

  function fillQueue() {
    while (queue.length < 3) queue.push(pull());
  }

  function collides(matrix, x, y) {
    for (let r = 0; r < matrix.length; r += 1) {
      for (let c = 0; c < matrix[r].length; c += 1) {
        if (!matrix[r][c]) continue;
        const bx = x + c;
        const by = y + r;
        if (bx < 0 || bx >= COLS || by >= ROWS) return true;
        if (by >= 0 && board[by][bx]) return true;
      }
    }
    return false;
  }

  function spawn(type) {
    piece = {
      type,
      matrix: cloneMatrix(SHAPES[type]),
      x: SPAWN_X[type],
      y: 0,
    };
    if (collides(piece.matrix, piece.x, piece.y)) status = "over";
  }

  function spawnNext() {
    fillQueue();
    spawn(queue.shift());
    if (status === "playing") fillQueue();
  }

  function start() {
    if (status === "playing" || status === "paused") return;
    if (status === "over") reset();
    status = "playing";
    spawnNext();
  }

  function restart() {
    reset();
    start();
  }

  function togglePause() {
    if (status === "playing") status = "paused";
    else if (status === "paused") status = "playing";
  }

  function move(dx, dy) {
    if (status !== "playing" || !piece) return false;
    const nx = piece.x + dx;
    const ny = piece.y + dy;
    if (collides(piece.matrix, nx, ny)) return false;
    piece.x = nx;
    piece.y = ny;
    return true;
  }

  function rotate(direction) {
    if (status !== "playing" || !piece) return false;
    const matrix = rotateMatrix(piece.matrix, direction);
    const kicks = [
      [0, 0],
      [-1, 0],
      [1, 0],
      [-2, 0],
      [2, 0],
      [0, -1],
      [-1, -1],
      [1, -1],
    ];
    for (const [kx, ky] of kicks) {
      if (!collides(matrix, piece.x + kx, piece.y + ky)) {
        piece.matrix = matrix;
        piece.x += kx;
        piece.y += ky;
        return true;
      }
    }
    return false;
  }

  function clearLines() {
    let cleared = 0;
    for (let y = ROWS - 1; y >= 0; ) {
      if (board[y].every(Boolean)) {
        board.splice(y, 1);
        board.unshift(Array(COLS).fill(null));
        cleared += 1;
      } else {
        y -= 1;
      }
    }
    return cleared;
  }

  function lock() {
    if (!piece) return;
    let above = false;
    for (let r = 0; r < piece.matrix.length; r += 1) {
      for (let c = 0; c < piece.matrix[r].length; c += 1) {
        if (!piece.matrix[r][c]) continue;
        const x = piece.x + c;
        const y = piece.y + r;
        if (y < 0) above = true;
        else board[y][x] = piece.type;
      }
    }
    const cleared = clearLines();
    lastClear = cleared;
    lastPoints = lineScore(cleared, level);
    score += lastPoints;
    lines += cleared;
    level = Math.floor(lines / 10) + 1;
    holdUsed = false;
    piece = null;
    if (above) {
      status = "over";
      return;
    }
    spawnNext();
  }

  function softDrop() {
    if (!move(0, 1)) return false;
    score += 1;
    accumulator = 0;
    return true;
  }

  function hardDrop() {
    if (status !== "playing" || !piece) return 0;
    let distance = 0;
    while (move(0, 1)) distance += 1;
    score += distance * 2;
    lock();
    return distance;
  }

  function hold() {
    if (status !== "playing" || !piece || holdUsed) return false;
    holdUsed = true;
    const current = piece.type;
    if (!holdType) {
      holdType = current;
      piece = null;
      spawnNext();
    } else {
      const swapped = holdType;
      holdType = current;
      spawn(swapped);
    }
    accumulator = 0;
    return status === "playing";
  }

  function tick(ms) {
    if (status !== "playing" || !piece) return;
    accumulator += Math.min(Math.max(ms, 0), 2000);
    const interval = dropInterval(level);
    while (accumulator >= interval && status === "playing" && piece) {
      accumulator -= interval;
      if (!move(0, 1)) lock();
    }
  }

  function activeCells(target = piece) {
    if (!target) return [];
    const cells = [];
    for (let r = 0; r < target.matrix.length; r += 1) {
      for (let c = 0; c < target.matrix[r].length; c += 1) {
        if (target.matrix[r][c]) cells.push({ x: target.x + c, y: target.y + r });
      }
    }
    return cells;
  }

  function ghostCells() {
    if (!piece || status === "over") return [];
    let y = piece.y;
    while (!collides(piece.matrix, piece.x, y + 1)) y += 1;
    return activeCells({ ...piece, y });
  }

  function getState() {
    return {
      status,
      score,
      lines,
      level,
      lastClear,
      lastPoints,
      hold: holdType,
      holdUsed,
      queue: queue.slice(),
      board: board.map((row) => row.slice()),
      piece: piece ? { type: piece.type, x: piece.x, y: piece.y } : null,
      active: activeCells(),
      ghost: ghostCells(),
    };
  }

  reset();

  return {
    start,
    restart,
    togglePause,
    move,
    rotate,
    softDrop,
    hardDrop,
    hold,
    tick,
    getState,
  };
}
