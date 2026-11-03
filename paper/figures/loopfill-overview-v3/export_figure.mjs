// Export the edited Obsidian drawing. Requires the Excalidraw 0.18 renderer,
// esbuild and playwright-core, plus a local Chromium (CHROMIUM_PATH).
import { createRequire } from "node:module";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const modules = process.env.LOOPFILL_RENDER_PACKAGES || here;
const require = createRequire(path.join(modules, "package.json"));
const { build } = require("esbuild");
const { chromium } = require("playwright-core");
const name = "loopfill-overview";
const markdown = await readFile(path.join(here, name + ".excalidraw.md"), "utf8");
const scene = JSON.parse(markdown.match(/## Drawing\s+```json\s+([\s\S]*?)\s+```/u)[1]);
const bundle = await build({
  stdin: { contents: [
    'import { restoreElements, exportToSvg } from "@excalidraw/excalidraw";',
    'window.exportScene = async scene => {',
    '  const elements = restoreElements(scene.elements, null, {repairBindings: true});',
    '  if (elements.length !== scene.elements.length) throw Error("Invalid elements on restore");',
    '  const svg = await exportToSvg({elements, appState: {...scene.appState,',
    '    exportBackground: true, exportWithDarkMode: false}, files: scene.files, exportPadding: 4});',
    '  return svg.outerHTML;',
    '};',
  ].join("\n"), resolveDir: modules },
  bundle: true, write: false, format: "iife", minify: true,
  define: {
    "process.env.NODE_ENV": '"production"',
    "import.meta.env": '{"PROD":true,"DEV":false,"MODE":"production"}',
  },
  loader: { ".woff2": "empty", ".css": "empty" },
});
const lib = path.dirname(require.resolve("@excalidraw/excalidraw"));
const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH});
try {
  const page = await browser.newPage({ deviceScaleFactor: 4 });
  // Keep the renderer offline. Serve the pinned library's own fonts from disk.
  await page.route("**/*", async route => {
    const { origin, pathname } = new URL(route.request().url());
    if (origin === "https://figure.invalid" && pathname === "/") {
      return route.fulfill({ contentType: "text/html", body: '<meta charset="utf-8">' });
    }
    if (origin === "https://figure.invalid" && pathname.startsWith("/fonts/") && pathname.endsWith(".woff2")) {
      return route.fulfill({ contentType: "font/woff2", body: await readFile(path.join(lib, pathname)) });
    }
    return route.abort();
  });
  await page.goto("https://figure.invalid/");
  await page.evaluate(() => { window.EXCALIDRAW_ASSET_PATH = "https://figure.invalid/"; });
  await page.addScriptTag({content: bundle.outputFiles[0].text});
  const svg = await page.evaluate(scene => window.exportScene(scene), scene);
  await page.evaluate(svg => { document.body.innerHTML = svg; }, svg);
  const view = await page.locator("svg").evaluate(s => [s.viewBox.baseVal.width, s.viewBox.baseVal.height]);
  const height = 3.25 * view[1] / view[0];
  await page.addStyleTag({content: `
    @page {size: 3.25in ${height}in; margin: 0}
    html,body {margin:0; padding:0; background:white}
    body>svg {display:block; width:3.25in; height:${height}in}
  `});
  await page.evaluate(async () => { await document.fonts.ready; });
  await page.locator("body>svg").screenshot({path:path.join(here,name + ".png")});
  await page.pdf({path:path.join(here,"../02_tree.pdf"), preferCSSPageSize:true, printBackground:true});
  await writeFile(path.join(here,name + ".svg"), svg);
  await writeFile(path.join(here,name + ".excalidraw"), JSON.stringify(scene,null,1) + "\n");
  console.log(`Restored ${scene.elements.length} elements from Markdown. Exported at 3.25 × ${height.toFixed(2)} in.`);
} finally {
  await browser.close();
}
