import assert from "node:assert/strict";
import test from "node:test";
import { createGame, dropInterval, lineScore } from "../js/engine.js";

test("줄 점수와 낙하 간격", () => {
  assert.equal(lineScore(1, 1), 100);
  assert.equal(lineScore(2, 1), 300);
  assert.equal(lineScore(3, 2), 1000);
  assert.equal(lineScore(4, 2), 1600);
  assert.equal(dropInterval(1), 800);
  assert.equal(dropInterval(2), 730);
  assert.equal(dropInterval(20), 100);
});

test("시작 전에는 조작이 무시된다", () => {
  const game = createGame({ sequence: ["T", "I", "O", "S"] });
  assert.equal(game.getState().status, "ready");
  assert.equal(game.move(-1, 0), false);
  assert.equal(game.hardDrop(), 0);
  game.tick(5000);
  assert.equal(game.getState().piece, null);
});

test("왼쪽 벽에서 더 이동하지 않고 T를 시계 방향으로 돌린다", () => {
  const game = createGame({ sequence: ["T", "I", "O", "S", "Z"] });
  game.start();
  for (let i = 0; i < 10; i += 1) game.move(-1, 0);
  assert.equal(game.getState().piece.x, 0);
  assert.equal(game.rotate(1), true);
  const cells = game.getState().active.map((cell) => `${cell.x},${cell.y}`).sort();
  assert.deepEqual(cells, ["1,0", "1,1", "1,2", "2,1"]);
});

test("중력은 간격마다 한 칸만 내린다", () => {
  const game = createGame({ sequence: ["T", "I", "O", "S", "Z"] });
  game.start();
  const y = game.getState().piece.y;
  game.tick(799);
  assert.equal(game.getState().piece.y, y);
  game.tick(1);
  assert.equal(game.getState().piece.y, y + 1);
});

test("일시정지 중에는 시간이 지나도 블록이 내려가지 않는다", () => {
  const game = createGame({ sequence: ["T", "I", "O", "S", "Z"] });
  game.start();
  const y = game.getState().piece.y;
  game.togglePause();
  game.tick(5000);
  assert.equal(game.getState().status, "paused");
  assert.equal(game.getState().piece.y, y);
  assert.equal(game.move(1, 0), false);
  game.togglePause();
  game.tick(800);
  assert.equal(game.getState().piece.y, y + 1);
});

test("소프트 드롭은 1점, 하드 드롭은 칸당 2점", () => {
  const game = createGame({ sequence: ["O", "I", "T", "S", "Z"] });
  game.start();
  assert.equal(game.softDrop(), true);
  assert.equal(game.getState().score, 1);
  assert.equal(game.getState().piece.y, 1);
  const distance = game.hardDrop();
  assert.equal(distance, 17);
  assert.equal(game.getState().score, 1 + 17 * 2);
  assert.equal(game.getState().piece.type, "I");
});

test("O 블록 다섯 개로 두 줄을 지우고 열 줄이면 레벨이 오른다", () => {
  const game = createGame({ sequence: Array(40).fill("O") });
  game.start();
  for (let set = 0; set < 5; set += 1) {
    for (const target of [0, 2, 4, 6, 8]) placeO(game, target);
  }
  const state = game.getState();
  assert.equal(state.lines, 10);
  assert.equal(state.level, 2);
  assert.equal(state.lastClear, 2);
  assert.equal(state.score, 5 * (5 * 18 * 2 + 300));
  assert.equal(state.board.every((row) => row.every((cell) => cell === null)), true);
});

test("홀드는 블록이 고정되기 전에 한 번만 된다", () => {
  const game = createGame({ sequence: ["T", "I", "O", "S", "Z", "J", "L"] });
  game.start();
  assert.equal(game.hold(), true);
  assert.equal(game.getState().hold, "T");
  assert.equal(game.getState().piece.type, "I");
  assert.equal(game.hold(), false);
  assert.equal(game.getState().piece.type, "I");
  game.hardDrop();
  assert.equal(game.getState().piece.type, "O");
  assert.equal(game.hold(), true);
  assert.equal(game.getState().hold, "O");
  assert.equal(game.getState().piece.type, "T");
});

test("7-bag은 일곱 종류를 한 번씩 꺼낸다", () => {
  let seed = 1;
  const random = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
  const game = createGame({ random });
  game.start();
  const seen = [];
  for (let i = 0; i < 7; i += 1) {
    seen.push(game.getState().piece.type);
    game.hardDrop();
    assert.equal(game.getState().status, "playing");
  }
  assert.equal(new Set(seen).size, 7);
});

test("꼭대기까지 쌓이면 게임 오버이고 다시 시작하면 점수가 0이다", () => {
  const game = createGame({ sequence: Array(30).fill("O") });
  game.start();
  let guard = 0;
  while (game.getState().status === "playing" && guard < 30) {
    game.hardDrop();
    guard += 1;
  }
  assert.equal(game.getState().status, "over");
  assert.ok(game.getState().score > 0);
  game.restart();
  assert.equal(game.getState().status, "playing");
  assert.equal(game.getState().score, 0);
  assert.equal(game.getState().lines, 0);
});

function placeO(game, targetX) {
  const startX = game.getState().piece.x;
  const steps = targetX - startX;
  for (let i = 0; i < Math.abs(steps); i += 1) game.move(Math.sign(steps), 0);
  assert.equal(game.getState().piece.x, targetX);
  game.hardDrop();
}
