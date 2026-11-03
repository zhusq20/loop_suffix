#!/usr/bin/env python3
"""LoopFill tree-construction figure (suffix tree + draft tree per block call).

Writes loopfill-draft-tree.excalidraw and loopfill-draft-tree.excalidraw.md
(Obsidian Excalidraw format). Export the PNG/SVG with export.mjs (see README).

The example matches the LoopFill panel of
results/loop-suffix-spec-20261030/figures/fig2-four-decoders.svg:
T = 3, readouts after loops 1 and 2, prompt "I would like",
output "to join Lean Club \\n" in 5 block calls.
"""
import json
import pathlib
import random
import string

OUT = pathlib.Path(__file__).resolve().parent
NAME = "loopfill-draft-tree"
rng = random.Random(20261101)

# Palette and type sizes copied from fig2-four-decoders.
FG, MUTED, RULE, OK, BAD = "#18202b", "#5a6573", "#d8dde4", "#2b7a4b", "#b33838"
LOOP = {1: ("#fbe8a6", "#c9a227"), 2: ("#c9e1f7", "#5e97cc"), 3: ("#c8edcc", "#4fa35d")}
TREE, FILL, BAND, IO, WHITE = "#2f6fd0", "#dd861b", "#f0f2f5", "#fbe3e3", "#ffffff"

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


def rect(x, y, w, h, stroke=FG, fill="transparent", dashed=False, sw=1, rounded=True):
    return base("rectangle", x, y, w, h, strokeColor=stroke, backgroundColor=fill,
                strokeStyle="dashed" if dashed else "solid", strokeWidth=sw,
                roundness={"type": 3} if rounded else None)


def text(s, x, yc, size=11, color=FG, align="left", glyph=False):
    """x is the left/center/right anchor; yc is the vertical center."""
    w = size if glyph else len(s) * 0.6 * size  # Cascadia advance = 0.6 em
    h = size * 1.25
    x0 = {"left": x, "center": x - w / 2, "right": x - w}[align]
    return base("text", x0, yc - h / 2, w, h, strokeColor=color, text=s, originalText=s,
                fontSize=size, fontFamily=3, textAlign=align, verticalAlign="top",
                containerId=None, lineHeight=1.25, autoResize=True)


def poly(points, color, head=True, kind="arrow", sw=1):
    if head:  # Excalidraw sizes the head by the last segment: keep that segment 12 px.
        (ax, ay), (bx, by) = points[-2], points[-1]
        d = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        if d > 14:
            points = points[:-1] + [(bx - (bx - ax) * 12 / d, by - (by - ay) * 12 / d), (bx, by)]
    x0, y0 = points[0]
    rel = [[round(px - x0, 2), round(py - y0, 2)] for px, py in points]
    xs, ys = [p[0] for p in rel], [p[1] for p in rel]
    return base(kind, x0, y0, max(xs) - min(xs), max(ys) - min(ys), strokeColor=color,
                strokeWidth=sw, points=rel, lastCommittedPoint=None, startBinding=None,
                endBinding=None, startArrowhead=None,
                endArrowhead="triangle" if head else None, elbowed=False)


def dot(cx, cy, r, color, fill=None):
    return base("ellipse", cx - r, cy - r, 2 * r, 2 * r, strokeColor=color,
                backgroundColor=fill or color)


def badge(n, cx, cy):
    dot(cx, cy, 6.5, TREE)
    text(str(n), cx, cy, 9, WHITE, "center")


def check(cx, cy):
    dot(cx, cy, 6.5, OK)
    text("✓", cx, cy, 9, WHITE, "center", glyph=True)


def scissors(cx, cy):
    text("✂", cx, cy, 13, BAD, "center", glyph=True)


NW, NH = 46, 18  # draft-tree node size


def node(x, yc, tok, loops=0, wrong=False, root=False, w=NW, fg=None, stroke=None):
    """loops: loops run so far (0 = added, not run yet). wrong: computed, then pruned."""
    y = yc - NH / 2
    if wrong:
        rect(x, y, w, NH, stroke=MUTED, dashed=True)
        color = BAD
    elif loops:
        fill, s = LOOP[loops]
        rect(x, y, w, NH, stroke=FG if root else s, fill=fill, sw=2 if root else 1)
        color = FG
    else:
        rect(x, y, w, NH, stroke=stroke or (FG if root else MUTED), fill=WHITE,
             sw=2 if root else 1)
        color = FG
    text(tok, x + w / 2, yc, 11, fg or color, "center")


def edge(px, pyc, cx, cyc, color, readout=False, w=NW):
    """Parent right side -> child left side; elbow edges drop in the column gap."""
    sx = px + w
    if abs(pyc - cyc) < 0.1:
        pts = [(sx, pyc), (cx - 1, cyc)]
    else:
        sy, mx = pyc + 5, sx + 8
        pts = [(sx, sy), (mx, sy), (mx, cyc), (cx - 1, cyc)]
    poly(pts, color)
    if readout:
        dot(pts[0][0], pts[0][1], 2.6, color)


# ---------------------------------------------------------------- header
rect(0, 4, 290, 52, stroke=RULE, fill=IO)
text("Input: I would like", 145, 20, 11, FG, "center")
text("Ground truth: to join Lean Club \\n", 145, 40, 11, FG, "center")

lx = 312
for k in (1, 2, 3):
    f, s = LOOP[k]
    rect(lx, 10, 47, 20, stroke=s, fill=f)
    text(f"loop{k}", lx + 23.5, 20, 11, FG, "center")
    lx += 55
rect(lx, 10, 60, 20, stroke=MUTED, fill=WHITE)
text("not run", lx + 30, 20, 11, FG, "center")
lx += 68
rect(lx, 10, 86.6, 20, stroke=MUTED, dashed=True)
text("wrong draft", lx + 43.3, 20, 11, FG, "center")
lx += 94.6
rect(lx, 10, 40, 20, stroke=FG, fill=WHITE, sw=2)
text("root", lx + 20, 20, 11, FG, "center")
lx += 56
check(lx, 20)
text("committed", lx + 10, 20, 10.5, MUTED)
lx += 84
scissors(lx, 20)
text("pruned", lx + 10, 20, 10.5, MUTED)

poly([(316, 41), (316, 53)], TREE)
text("draft from suffix tree", 330, 47, 10.5, MUTED)
dot(507, 42, 2.6, FILL)
poly([(507, 42), (507, 50), (521, 50)], FILL)
text("draft from shallow loop", 527, 47, 10.5, MUTED)
dot(706, 42, 2.6, FG)
poly([(706, 42), (706, 50), (720, 50)], FG)
text("final (loop-3) prediction", 726, 47, 10.5, MUTED)

# ------------------------------------------------- (a) suffix tree panel
text("(a) Suffix tree over earlier text", 0, 82, 13, FG)
text('earlier output: "Would you like a coffee"', 0, 106, 10.5, MUTED)
text('earlier output: "We want to join Lean Hub"', 0, 122, 10.5, MUTED)

SP = 58  # column pitch in the suffix tree
SX = [34 + SP * i for i in range(4)]
yA, yB, yC = 168, 206, 244
dot(12, yB, 5, MUTED, WHITE)
for y in (yA, yB, yC):
    poly([(17, yB), (24, yB), (24, y), (33, y)] if y != yB else [(17, yB), (33, yB)],
         MUTED, head=False, kind="line")
node(SX[0], yA, "like")
for i, tok in enumerate(["a", "coffee"], start=1):
    node(SX[i], yA, tok, stroke=TREE, fg=TREE)
    poly([(SX[i - 1] + NW, yA), (SX[i] - 1, yA)], TREE)
node(SX[0], yB, "to")
for i, tok in enumerate(["join", "Lean", "Hub"], start=1):
    node(SX[i], yB, tok, stroke=TREE, fg=TREE)
    poly([(SX[i - 1] + NW, yB), (SX[i] - 1, yB)], TREE)
text("… other suffixes", SX[0], yC, 10.5, MUTED)
badge(1, SX[0] - 2, yA - NH / 2 - 2)
badge(2, SX[0] - 2, yB - NH / 2 - 2)

badge(1, 6.5, 294)
text("match like → draft a coffee", 18, 294, 10.5, FG)
text("(before call 1)", 18, 309, 10.5, MUTED)
badge(2, 6.5, 332)
text("match to → draft join Lean Hub", 18, 332, 10.5, FG)
text("(after call 1 adds to)", 18, 347, 10.5, MUTED)

# ------------------------------------------------ (b) draft-tree panel
text("(b) Draft tree after each block call", 312, 82, 13, FG)
text("T = 3 loops; shallow readouts after loops 1 and 2", 312, 98, 10.5, MUTED)

X = [368 + 72 * i for i in range(6)]  # columns = output positions
LP = 26  # line pitch inside a row
ANN = 800


def row(top, call, nlines, notes, band=False):
    h = NH + LP * (nlines - 1)
    if band:
        rect(302, top - 8, 748, h + 16, stroke="transparent", fill=BAND, rounded=False)
    ys = [top + NH / 2 + LP * i for i in range(nlines)]
    text(f"call {call}", 312, ys[0], 11.5, FG)
    mid = top + h / 2
    for i, s in enumerate(notes):
        text(s, ANN, mid + (i - (len(notes) - 1) / 2) * 13, 10, MUTED)
    return ys, top + h + 18


top = 118
# call 1: root runs loop 1 together with the suffix draft "a coffee".
ys, nxt = row(top, 1, 2, ["like, a, coffee run loop 1.",
                          "Readout at like ranks to first:",
                          "add to, prune a coffee.",
                          "Suffix match on to adds",
                          "join Lean Hub."])
l0, l1 = ys
node(X[0], l0, "like", 1, root=True)
node(X[1], l1, "a", wrong=True)
edge(X[0], l0, X[1], l1, TREE)
node(X[2], l1, "coffee", wrong=True)
edge(X[1], l1, X[2], l1, TREE)
scissors(X[2] + NW + 11, l1)
badge(1, X[1] - 2, l1 - NH / 2 - 2)
node(X[1], l0, "to")
edge(X[0], l0, X[1], l0, FILL, readout=True)
node(X[2], l0, "join")
edge(X[1], l0, X[2], l0, TREE)
badge(2, X[2] - 2, l0 - NH / 2 - 2)
node(X[3], l0, "Lean")
edge(X[2], l0, X[3], l0, TREE)
node(X[4], l1, "Hub")
edge(X[3], l0, X[4], l1, TREE)

# call 2: loop-1 readouts at join and Lean add "the" and "Club"; "Hub" is pruned.
ys, nxt = row(nxt, 2, 3, ["to, join, Lean, Hub run loop 1.",
                          "Readout at join adds the.",
                          "Readout at Lean adds Club",
                          "and prunes Hub."], band=True)
l0, l1, l2 = ys
node(X[0], l0, "like", 2, root=True)
node(X[1], l0, "to", 1)
edge(X[0], l0, X[1], l0, FILL, readout=True)
node(X[2], l0, "join", 1)
edge(X[1], l0, X[2], l0, TREE)
node(X[3], l0, "Lean", 1)
edge(X[2], l0, X[3], l0, TREE)
node(X[4], l0, "Club")
edge(X[3], l0, X[4], l0, FILL, readout=True)
node(X[3], l2, "the")
edge(X[2], l0, X[3], l2, FILL, readout=True)
node(X[4], l1, "Hub", wrong=True)
edge(X[3], l0, X[4], l1, TREE)
scissors(X[4] + NW + 11, l1)

# call 3: the root finishes loop 3 and commits "to"; the loop-2 readout at join prunes "the".
ys, nxt = row(nxt, 3, 2, ["like finishes loop 3: commit to.",
                          "Loop-2 readout at join",
                          "ranks Lean first: prune the."])
l0, l1 = ys
node(X[0], l0, "like", 3, root=True)
node(X[1], l0, "to", 2)
edge(X[0], l0, X[1], l0, FILL, readout=True)
check(X[1] + NW, l0 - NH / 2 - 3)
node(X[2], l0, "join", 2)
edge(X[1], l0, X[2], l0, TREE)
node(X[3], l0, "Lean", 2)
edge(X[2], l0, X[3], l0, TREE)
node(X[4], l0, "Club", 1)
edge(X[3], l0, X[4], l0, FILL, readout=True)
node(X[3], l1, "the", wrong=True)
edge(X[2], l0, X[3], l1, FILL, readout=True)
scissors(X[3] + NW + 11, l1)

# call 4: three nodes finish loop 3 in the same call; three tokens are committed.
ys, nxt = row(nxt, 4, 1, ["to, join, Lean finish loop 3:",
                          "commit join, Lean, Club."], band=True)
l0 = ys[0]
text("like", X[0] + NW / 2, l0, 11, MUTED, "center")
node(X[1], l0, "to", 3, root=True)
for i, (tok, k) in enumerate([("join", 3), ("Lean", 3), ("Club", 2)], start=2):
    node(X[i], l0, tok, k)
    edge(X[i - 1], l0, X[i], l0, FILL if tok == "Club" else TREE, readout=tok == "Club")
    check(X[i] + NW, l0 - NH / 2 - 3)

# call 5: the last node finishes loop 3.
ys, nxt = row(nxt, 5, 1, ["Club finishes loop 3: commit \\n.",
                          "5 calls; greedy decoding needs 15."])
l0 = ys[0]
for i, tok in enumerate(["like", "to", "join", "Lean"]):
    text(tok, X[i] + NW / 2, l0, 11, MUTED, "center")
node(X[4], l0, "Club", 3, root=True)
text("\\n", X[5] + 12, l0, 11, FG, "center")
dot(X[4] + NW, l0, 2.6, FG)
poly([(X[4] + NW, l0), (X[5] + 3, l0)], FG)
check(X[5] + 30, l0)
poly([(298, 70), (298, nxt - 10)], RULE, head=False, kind="line")

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
