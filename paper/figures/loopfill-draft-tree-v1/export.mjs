// Usage: node export.mjs <scene.excalidraw> <out.png> <out.svg> [scale] [padding]
import http from "node:http"; import fs from "node:fs"; import path from "node:path";
import { chromium } from "playwright-core";
const here = path.dirname(new URL(import.meta.url).pathname);
const [scenePath, pngPath, svgPath, scaleArg, padArg] = process.argv.slice(2);
const scene = JSON.parse(fs.readFileSync(scenePath, "utf8"));
const types = { ".js": "text/javascript", ".html": "text/html", ".woff2": "font/woff2", ".css": "text/css" };
const server = http.createServer((req, res) => {
  const p = path.join(here, decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (!p.startsWith(here) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "content-type": types[path.extname(p)] || "application/octet-stream" });
  fs.createReadStream(p).pipe(res);
});
await new Promise(r => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;
fs.writeFileSync(path.join(here, "index.html"), `<!doctype html><meta charset="utf-8"><script>window.EXCALIDRAW_ASSET_PATH="${base}/node_modules/@excalidraw/excalidraw/dist/prod/";</script><script src="/bundle.js"></script>`);
const exe = fs.readdirSync(path.join(process.env.HOME, ".cache/ms-playwright/chromium-1234")).filter(d => d.startsWith("chrome-linux"))[0];
const browser = await chromium.launch({ executablePath: path.join(process.env.HOME, ".cache/ms-playwright/chromium-1234", exe, "chrome") });
const page = await browser.newPage();
page.on("console", m => { if (m.type() === "error") console.error("[page]", m.text()); });
await page.goto(`${base}/index.html`);
await page.waitForFunction(() => typeof window.runExport === "function");
const out = await page.evaluate(([s, sc, pd]) => window.runExport(s, sc, pd), [scene, Number(scaleArg || 3), Number(padArg || 12)]);
fs.writeFileSync(pngPath, Buffer.from(out.png, "base64"));
fs.writeFileSync(svgPath, out.svg);
await browser.close(); server.close();
console.log("wrote", pngPath, svgPath);
