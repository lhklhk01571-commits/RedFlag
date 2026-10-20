import { createGame } from "./engine.js";
import { drawBoard, drawPiece } from "./draw.js";

const BEST_KEY = "tetris-best-score";
const CLEAR_NAMES = ["", "싱글", "더블", "트리플", "테트리스"];

const game = createGame();
const boardCanvas = document.querySelector("#board");
const holdCanvas = document.querySelector("#hold");
const nextCanvas = document.querySelector("#next");
const boardCtx = boardCanvas.getContext("2d");
const holdCtx = holdCanvas.getContext("2d");
const nextCtx = nextCanvas.getContext("2d");
const scoreEl = document.querySelector("#score");
const bestEl = document.querySelector("#best");
const levelEl = document.querySelector("#level");
const linesEl = document.querySelector("#lines");
const bannerEl = document.querySelector("#banner");
const overlayEl = document.querySelector("#overlay");
const overlayTitle = document.querySelector("#overlay-title");
const overlayText = document.querySelector("#overlay-text");
const actionButton = document.querySelector("#action");
const appEl = document.querySelector(".app");

let best = readBest();
let seenLines = 0;
let bannerUntil = 0;
let last = performance.now();
bestEl.textContent = String(best);

fitCanvases();
window.addEventListener("resize", fitCanvases);

actionButton.addEventListener("click", () => {
  const { status } = game.getState();
  if (status === "paused") game.togglePause();
  else if (status === "over") game.restart();
  else game.start();
  actionButton.blur();
});

document.addEventListener("keydown", (event) => {
  const action = KEYS[event.code];
  if (!action) return;
  if (event.repeat && (action === "rot" || action === "drop" || action === "hold" || action === "pause")) return;
  event.preventDefault();
  run(action);
});

for (const button of document.querySelectorAll("[data-action]")) {
  let timer = 0;
  const action = button.dataset.action;
  const stop = () => {
    window.clearInterval(timer);
    timer = 0;
  };
  button.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    run(action);
    if (action === "left" || action === "right" || action === "down") {
      timer = window.setInterval(() => run(action), 110);
    }
  });
  button.addEventListener("pointerup", stop);
  button.addEventListener("pointerleave", stop);
  button.addEventListener("pointercancel", stop);
}

requestAnimationFrame(frame);

function frame(now) {
  const dt = Math.min(now - last, 100);
  last = now;
  game.tick(dt);
  render(now);
  requestAnimationFrame(frame);
}

function run(action) {
  const { status } = game.getState();
  if (status === "ready") return;
  if (action === "pause") {
    game.togglePause();
    return;
  }
  if (status !== "playing") return;
  if (action === "left") game.move(-1, 0);
  if (action === "right") game.move(1, 0);
  if (action === "rot") game.rotate(1);
  if (action === "ccw") game.rotate(-1);
  if (action === "down") game.softDrop();
  if (action === "drop") game.hardDrop();
  if (action === "hold") game.hold();
}

function render(now) {
  const state = game.getState();
  if (state.score > best) {
    best = state.score;
    try {
      localStorage.setItem(BEST_KEY, String(best));
    } catch {
      // 저장소를 쓸 수 없어도 이번 판은 계속한다.
    }
  }
  if (state.lines > seenLines && state.lastClear) {
    bannerEl.textContent = `${CLEAR_NAMES[state.lastClear]} +${state.lastPoints}`;
    bannerUntil = now + 700;
  }
  seenLines = state.lines;
  bannerEl.hidden = now > bannerUntil;

  scoreEl.textContent = String(state.score);
  bestEl.textContent = String(best);
  levelEl.textContent = String(state.level);
  linesEl.textContent = String(state.lines);
  appEl.dataset.status = state.status;

  drawBoard(boardCtx, state, boardCanvas.width / dpr(), boardCanvas.height / dpr());
  drawPiece(holdCtx, state.hold, holdCanvas.width / dpr(), holdCanvas.height / dpr());
  drawNext(state.queue);

  if (state.status === "playing") {
    overlayEl.hidden = true;
    return;
  }
  overlayEl.hidden = false;
  if (state.status === "paused") {
    overlayTitle.textContent = "일시정지";
    overlayText.textContent = "이어서 블록을 내려 보세요.";
    actionButton.textContent = "계속하기";
  } else if (state.status === "over") {
    overlayTitle.textContent = "게임 오버";
    overlayText.textContent = `점수 ${state.score}점`;
    actionButton.textContent = "다시 하기";
  } else {
    overlayTitle.textContent = "테트리스";
    overlayText.textContent = "줄을 채워 지우면 점수가 오릅니다.";
    actionButton.textContent = "시작";
  }
}

function drawNext(queue) {
  const width = nextCanvas.width / dpr();
  const height = nextCanvas.height / dpr();
  nextCtx.clearRect(0, 0, width, height);
  const slot = height / 3;
  queue.slice(0, 3).forEach((type, index) => {
    nextCtx.save();
    nextCtx.translate(0, index * slot);
    nextCtx.beginPath();
    nextCtx.rect(0, 0, width, slot);
    nextCtx.clip();
    drawPiece(nextCtx, type, width, slot);
    nextCtx.restore();
  });
}

function fitCanvases() {
  const ratio = dpr();
  sizeCanvas(boardCanvas, boardCtx, 300, 600, ratio);
  sizeCanvas(holdCanvas, holdCtx, 112, 96, ratio);
  sizeCanvas(nextCanvas, nextCtx, 112, 280, ratio);
}

function sizeCanvas(canvas, ctx, width, height, ratio) {
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function dpr() {
  return Math.min(window.devicePixelRatio || 1, 2);
}

function readBest() {
  try {
    const value = Number(localStorage.getItem(BEST_KEY) || 0);
    return Number.isFinite(value) ? value : 0;
  } catch {
    return 0;
  }
}

const KEYS = {
  ArrowLeft: "left",
  ArrowRight: "right",
  ArrowDown: "down",
  ArrowUp: "rot",
  KeyX: "rot",
  KeyZ: "ccw",
  Space: "drop",
  KeyC: "hold",
  KeyP: "pause",
};
