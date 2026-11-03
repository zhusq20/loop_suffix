#!/usr/bin/env python3
"""Compact LoopFill tree figure (single column): suffix tree + one draft tree.

The per-step timeline is left to fig2-four-decoders (LoopFill panel). This
figure only shows what fig2 cannot: where each branch came from and which
step accepted or pruned it. Same example as fig2: T = 3, prompt "I would like",
output "to join Lean Club \\n" in 5 loop steps.

Writes loopfill-draft-tree.excalidraw and loopfill-draft-tree.excalidraw.md.
Export PNG/SVG with ../loopfill-draft-tree-v1/export.mjs (see README).
"""
import json
import pathlib
import random
import string

OUT = pathlib.Path(__file__).resolve().parent
NAME = "loopfill-draft-tree"
rng = random.Random(20261101)

# Palette copied from fig2-four-decoders.
FG, MUTED, OK, BAD = "#18202b", "#5a6573", "#2b7a4b", "#b33838"
TREE, FILL, WHITE = "#2f6fd0", "#dd861b", "#ffffff"

els = []


def _id():
    return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(16))


def base(kind, x, y, w, h, **kw):
    e = dict(id=_id(), type=kind, x=round(x, 2), y=round(y, 2), width=round(w, 2),
             height=round(h, 2), angle=0, strokeColor=FG, backgroundColor="transparent",
             fillStyle="solid", strokeWidth=1, strokeStyle="solid", roughness=0, opacity=100,
             groupIds=[], frameId=None, roundness=None, seed=rng.randrange(1, 2**31),
             version=1, versionNonce=rng.randrange(1, 2**31), isDeleted=False,
             boundElements=None, updated=1793520000000, link=None, locked=False)
    e.update(kw)
    els.append(e)
    return e


def rect(x, y, w, h, stroke=FG, fill="transparent", dashed=False, sw=1):
    return base("rectangle", x, y, w, h, strokeColor=stroke, backgroundColor=fill,
                strokeStyle="dashed" if dashed else "solid", strokeWidth=sw,
                roundness={"type": 3})


def text(s, x, yc, size=11, color=FG, align="left", width=None):
    """x is the left/center anchor, yc the vertical center."""
    w = width if width is not None else len(s) * 0.6 * size  # Cascadia: 0.6 em
    h = size * 1.25
    x0 = x - w / 2 if align == "center" else x
    return base("text", x0, yc - h / 2, w, h, strokeColor=color, text=s, originalText=s,
                fontSize=size, fontFamily=3, textAlign=align, verticalAlign="top",
                containerId=None, lineHeight=1.25, autoResize=True)


def poly(points, color, head=True, kind="arrow"):
    if head:  # Excalidraw sizes the head by the last segment: keep it 12 units.
        (ax, ay), (bx, by) = points[-2], points[-1]
        d = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        if d > 14:
            points = points[:-1] + [(bx - (bx - ax) * 12 / d, by - (by - ay) * 12 / d), (bx, by)]
    x0, y0 = points[0]
    rel = [[round(px - x0, 2), round(py - y0, 2)] for px, py in points]
    xs, ys = [p[0] for p in rel], [p[1] for p in rel]
    return base(kind, x0, y0, max(xs) - min(xs), max(ys) - min(ys), strokeColor=color,
                points=rel, lastCommittedPoint=None, startBinding=None, endBinding=None,
                startArrowhead=None, endArrowhead="triangle" if head else None,
                elbowed=False)


def dot(cx, cy, r, color, fill=None):
    return base("ellipse", cx - r, cy - r, 2 * r, 2 * r, strokeColor=color,
                backgroundColor=fill or color)


def badge(label, cx, cy, color):
    dot(cx, cy, 6.5, color)
    text(str(label), cx, cy, 9, WHITE, "center")


def pruned(step, x, yc):
    text(f"✂{step}", x, yc, 11, BAD, width=2 * 0.6 * 11 + 4)


NW, NH = 44, 18


def node(x, yc, tok, wrong=False, root=False, suffix=False):
    color = BAD if wrong else (TREE if suffix else FG)
    stroke = MUTED if wrong else (FG if root else (TREE if suffix else MUTED))
    rect(x, yc - NH / 2, NW, NH, stroke=stroke, fill=WHITE, dashed=wrong, sw=2 if root else 1)
    text(tok, x + NW / 2, yc, 11, color, "center")


def edge(px, pyc, cx, cyc, color, shallow=False):
    """Parent right side -> child left side. Elbows turn in the column gap."""
    sx = px + NW
    if abs(pyc - cyc) < 0.1:
        pts = [(sx, pyc), (cx - 1, cyc)]
    else:
        sy = pyc + (5 if cyc > pyc else -5)
        pts = [(sx, sy), (sx + 8, sy), (sx + 8, cyc), (cx - 1, cyc)]
    poly(pts, color)
    if shallow:
        dot(pts[0][0], pts[0][1], 2.6, color)


# ---------------------------------------------------------------- legend
poly([(0, 8), (16, 8)], TREE)
text("from suffix tree", 21, 8, 10, MUTED)
dot(132, 8, 2.6, FILL)
poly([(132, 8), (148, 8)], FILL)
text("from shallow loop", 153, 8, 10, MUTED)
rect(0, 19, 24, 14, stroke=MUTED, dashed=True)
pruned("k", 30, 26)
text("pruned after step k", 52, 26, 10, MUTED)
badge("k", 184, 26, OK)
text("accepted at step k", 194, 26, 10, MUTED)

# ------------------------------------------------------ (a) suffix tree
text("(a) Suffix tree", 0, 54, 12, FG)
text("over earlier outputs", 116, 54, 10, MUTED)
text('"Would you like a coffee", "We want to join Lean Hub"', 0, 72, 10, MUTED)
SX = [22 + 56 * i for i in range(4)]
yA, yB, yC = 100, 126, 150
dot(8, yB, 4.5, MUTED, WHITE)
for y in (yA, yB, yC):
    pts = [(12.5, yB), (21, yB)] if y == yB else [(12.5, yB), (16, yB), (16, y), (21, y)]
    poly(pts, MUTED, head=False, kind="line")
node(SX[0], yA, "like")
for i, tok in enumerate(["a", "coffee"], start=1):
    node(SX[i], yA, tok, suffix=True)
    poly([(SX[i - 1] + NW, yA), (SX[i] - 1, yA)], TREE)
node(SX[0], yB, "to")
for i, tok in enumerate(["join", "Lean", "Hub"], start=1):
    node(SX[i], yB, tok, suffix=True)
    poly([(SX[i - 1] + NW, yB), (SX[i] - 1, yB)], TREE)
text("… other suffixes", SX[0], yC, 10, MUTED)
badge(1, SX[0] - 2, yA - NH / 2 - 2, TREE)
badge(2, SX[0] - 2, yB - NH / 2 - 2, TREE)

# ------------------------------------------------------ (b) draft tree
text("(b) Draft tree", 0, 180, 12, FG)
text('prompt "I would like", T = 3', 116, 180, 10, MUTED)
X = [6 + 62 * i for i in range(6)]  # columns = output positions
lm, l0, l1, l2 = 210, 240, 264, 288

node(X[0], l0, "like", root=True)
# suffix draft from match 1, pruned after step 1
node(X[1], lm, "a", wrong=True)
edge(X[0], l0, X[1], lm, TREE)
node(X[2], lm, "coffee", wrong=True)
edge(X[1], lm, X[2], lm, TREE)
badge(1, X[1] - 2, lm - NH / 2 - 2, TREE)
pruned(1, X[2] + NW + 6, lm)
# accepted path
node(X[1], l0, "to")
edge(X[0], l0, X[1], l0, FILL, shallow=True)
node(X[2], l0, "join")
edge(X[1], l0, X[2], l0, TREE)
badge(2, X[2] - 9, l0 + 14, TREE)
node(X[3], l0, "Lean")
edge(X[2], l0, X[3], l0, TREE)
node(X[4], l0, "Club")
edge(X[3], l0, X[4], l0, FILL, shallow=True)
text("\\n", X[5] + 8, l0, 11, FG, "center")
poly([(X[4] + NW, l0), (X[5] - 1, l0)], FG)
for i, step in [(1, 3), (2, 4), (3, 4), (4, 4)]:
    badge(step, X[i] + NW, l0 - NH / 2 - 2, OK)
badge(5, X[5] + 24, l0, OK)
# pruned side branches
node(X[4], l1, "Hub", wrong=True)
edge(X[3], l0, X[4], l1, TREE)
pruned(2, X[4] + NW + 6, l1)
node(X[3], l2, "the", wrong=True)
edge(X[2], l0, X[3], l2, FILL, shallow=True)
pruned(3, X[3] + NW + 6, l2)

# ------------------------------------------------------------- write
scene = {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com",
         "elements": els, "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
         "files": {}}
body = json.dumps(scene, indent=1, ensure_ascii=False)
(OUT / f"{NAME}.excalidraw").write_text(body + "\n")
texts = "\n".join(f"{e['text']} ^{e['id']}\n" for e in els if e["type"] == "text")
md = ("---\n\nexcalidraw-plugin: parsed\ntags: [excalidraw]\n\n---\n"
      "==⚠  Switch to EXCALIDRAW VIEW in the MORE OPTIONS menu of this document. ⚠==\n\n\n"
      "# Excalidraw Data\n\n## Text Elements\n" + texts +
      "\n%%\n## Drawing\n```json\n" + body + "\n```\n%%\n")
(OUT / f"{NAME}.excalidraw.md").write_text(md)
print(f"{len(els)} elements -> {OUT / NAME}.excalidraw(.md)")
