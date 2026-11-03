# LoopFill draft-tree figure (v1)

Shows how LoopFill builds its suffix tree and draft tree. It uses the same example as the
LoopFill panel of `results/loop-suffix-spec-20261030/figures/fig2-four-decoders.svg`:
T = 3, readouts after loops 1 and 2, prompt "I would like", output "to join Lean Club \n" in 5 calls.
The two "earlier output" sentences in panel (a) are illustrative.

- `loopfill-draft-tree.excalidraw.md`: Obsidian Excalidraw file (open in Excalidraw view)
- `loopfill-draft-tree.excalidraw`: the same scene as plain Excalidraw JSON
- `loopfill-draft-tree.png` (3x) and `.svg`: exported by Excalidraw itself

Rebuild:

```bash
python3 build_figure.py   # writes .excalidraw and .excalidraw.md
# PNG/SVG export through Excalidraw's own renderer in headless Chromium:
npm install @excalidraw/excalidraw@0.18.0 react@18.3.1 react-dom@18.3.1 esbuild@0.24.0 playwright-core@1.48.2
npx esbuild entry.js --bundle --format=iife --minify \
  --define:process.env.NODE_ENV='"production"' \
  --define:import.meta.env='{"PROD":true,"DEV":false,"MODE":"production"}' \
  --loader:.woff2=empty --loader:.css=empty --outfile=bundle.js
node export.mjs loopfill-draft-tree.excalidraw loopfill-draft-tree.png loopfill-draft-tree.svg 3 12
```

`export.mjs` launches the Chromium in `~/.cache/ms-playwright/chromium-1234`.
