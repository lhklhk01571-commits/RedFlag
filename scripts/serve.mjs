import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const preferred = Number(process.env.PORT) || 4173;
const types = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
};

function createApp() {
  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url ?? "/", "http://127.0.0.1");
      const relative = decodeURIComponent(url.pathname).replace(/^\/+/, "");
      const filePath = normalize(join(root, relative.endsWith("/") || relative === "" ? `${relative}index.html` : relative));
      const rootPrefix = root.endsWith(sep) ? root : `${root}${sep}`;
      if (filePath !== root && !filePath.startsWith(rootPrefix)) {
        response.writeHead(403).end("Forbidden");
        return;
      }
      const body = await readFile(filePath);
      const type = types[extname(filePath)] ?? "application/octet-stream";
      response.writeHead(200, { "Content-Type": type }).end(body);
    } catch (error) {
      const missing = error && error.code === "ENOENT";
      response.writeHead(missing ? 404 : 500).end(missing ? "Not found" : "Server error");
    }
  });
}

function listen(port) {
  return new Promise((resolve, reject) => {
    const server = createApp();
    server.once("error", reject);
    server.listen(port, "0.0.0.0", () => resolve(server));
  });
}

const first = Number.isInteger(preferred) && preferred > 0 ? preferred : 4173;
let started = false;
for (let port = first; port < first + 20; port += 1) {
  try {
    await listen(port);
    if (port !== first) console.log(`${first} 포트는 사용 중입니다.`);
    console.log(`테트리스 서버: http://127.0.0.1:${port}/`);
    started = true;
    break;
  } catch (error) {
    if (error.code !== "EADDRINUSE") throw error;
  }
}

if (!started) {
  console.error(`${first}부터 ${first + 19}까지 사용할 수 있는 포트가 없습니다.`);
  process.exit(1);
}
