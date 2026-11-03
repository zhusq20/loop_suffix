#!/usr/bin/env python3
"""Compose Figure 2 from the native elements in the original Excalidraw Markdown."""

import copy
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
ORIGINAL = OUT.parent / "loopfill-draft-tree-v2" / "loopfill-draft-tree.excalidraw.md"
original_md = ORIGINAL.read_text()
original = json.loads(original_md.split("```json\n")[1].split("\n```")[0])
source = original["elements"]
elements = []
FG, MUTED, OK, BAD = "#18202b", "#5a6573", "#2b7a4b", "#b33838"
TREE, SHALLOW = "#2f6fd0", "#dd861b"
WIDTH = 298


def sample(kind, **attrs):
    return next(e for e in source if e["type"] == kind and all(e[k] == v for k, v in attrs.items()))


def add(prototype, **fields):
    item = copy.deepcopy(prototype)
    item.update(id=f"lf{len(elements) + 1:06}", **fields)
    elements.append(item)
    return item


def label(value, x, cy, size=11, color=FG, align="left", group=None, font=2):
    # Use sans serif for explanations and the original Cascadia face for literal tokens.
    w = len(value) * (0.6 if font == 3 else 0.53) * size
    left = x - (w / 2 if align == "center" else w if align == "right" else 0)
    return add(
        sample("text", text="like"),
        x=left,
        y=cy - size * 0.625,
        width=w,
        height=size * 1.25,
        text=value,
        originalText=value,
        strokeColor=color,
        fontSize=size,
        fontFamily=font,
        textAlign=align,
        groupIds=[group] if group else [],
    )


def header(letter, title, step, y):
    label(f"({letter}) {title}", 0, y, 13)
    label(step, WIDTH, y, 10.6, MUTED, "right")


def mark(cx, cy, digit, color):
    group = f"badge{len(elements)}"
    proto = sample("ellipse", strokeColor=color, width=13)
    add(proto, x=cx - 7, y=cy - 7, width=14, height=14, groupIds=[group])
    label(digit, cx, cy, 10.3, "#ffffff", "center", group, font=3)


def line(points, color=MUTED, head=False, group=None):
    template = sample("arrow" if head else "line")
    if head:
        (ax, ay), (bx, by) = points[-2:]
        distance = math.hypot(bx - ax, by - ay)
        # Excalidraw sizes the triangle from the final segment, as in v2.
        if distance > 14:
            points = [
                *points[:-1],
                (bx - 12 * (bx - ax) / distance, by - 12 * (by - ay) / distance),
                (bx, by),
            ]
    x, y = points[0]
    xs, ys = zip(*points, strict=True)
    return add(
        template,
        x=x,
        y=y,
        width=max(xs) - min(xs),
        height=max(ys) - min(ys),
        points=[[px - x, py - y] for px, py in points],
        strokeColor=color,
        groupIds=[group] if group else [],
    )


def rule(y):
    line([(0, y), (WIDTH, y)], "#c9cfd5")["strokeWidth"] = 0.65


node_styles = {
    "root": sample("rectangle", width=44, strokeColor=FG),
    "live": sample("rectangle", width=44, strokeStyle="solid", strokeColor=MUTED),
    "suffix": sample("rectangle", width=44, strokeColor=TREE),
    "pruned": sample("rectangle", width=44, strokeStyle="dashed"),
}


def token(x, cy, word, style="live"):
    group = f"node{len(elements)}"
    box = add(node_styles[style], x=x, y=cy - 9, groupIds=[group])
    label(
        word,
        x + 22,
        cy,
        11.3,
        {"suffix": TREE, "pruned": BAD}.get(style, FG),
        "center",
        group,
        font=3,
    )
    return box


def connect(parent, child, color):
    x, y = parent["x"] + 44, parent["y"] + 9
    x2, y2 = child["x"] - 1, child["y"] + 9
    offset = 0 if y == y2 else (4.5 if y2 > y else -4.5)
    points = [(x, y + offset), (x2, y2)]
    if offset:
        elbow = (x + x2) / 2
        points[1:1] = [(elbow, y + offset), (elbow, y2)]
    group = f"edge{len(elements)}" if color == SHALLOW else None
    arrow = line(points, color, head=True, group=group)
    # Native bindings let a user move a token without detaching its arrows.
    for field, node, pin in (
        ("startBinding", parent, [1, (9 + offset) / 18]),
        ("endBinding", child, [0, 0.5]),
    ):
        arrow[field] = dict(elementId=node["id"], focus=0, gap=1, fixedPoint=pin)
        node["boundElements"] = (node["boundElements"] or []) + [dict(id=arrow["id"], type="arrow")]
    if color == SHALLOW:
        dot = sample("ellipse", width=5.2, strokeColor=SHALLOW)
        add(dot, x=x - 2.6, y=y + offset - 2.6, groupIds=[group])


# (a) Copy the suffix-tree geometry, token labels and lookup badges directly from v2.
header("a", "Draft from suffix matches", "Step 1", 9)
start = source.index(sample("ellipse", width=9.0))
end = source.index(sample("text", text="(b) Draft tree"))
for e in source[start:end]:
    # Show only the two matched branches of the suffix index.
    if e.get("text") == "… other suffixes" or (e["type"] == "line" and e["height"] == 24):
        continue
    add(e, x=e["x"] + 15, y=e["y"] - 62)
# Group copied tokens and badges with their own editable text.
for shape, text in zip(elements[:-1], elements[1:], strict=True):
    if shape["type"] in {"rectangle", "ellipse"} and text["type"] == "text":
        shape["groupIds"] = text["groupIds"] = [f"pair{shape['id']}"]
label("The three positions on path 1 start together.", 0, 83, 11, MUTED)
rule(94)

# (b) Show where each proposal comes from. Panel (c) shows pruning separately.
header("b", "Draft from early predictions", "Steps 1–2", 107)
line([(1, 123), (17, 123)], TREE, head=True)
label("suffix match", 22, 123, 10.8, MUTED)
add(sample("ellipse", width=5.2, strokeColor=SHALLOW), x=126 - 2.6, y=123 - 2.6)
line([(126, 123), (142, 123)], SHALLOW, head=True)
label("early prediction", 147, 123, 10.8, MUTED)

like = token(1, 145, "like", "root")
to = token(59, 145, "to")
join = token(117, 145, "join")
lean = token(175, 145, "Lean")
club = token(233, 145, "Club")
the = token(175, 169, "the")
hub = token(233, 169, "Hub")
for parent, child, color in [
    (like, to, SHALLOW),
    (to, join, TREE),
    (join, lean, TREE),
    (lean, club, SHALLOW),
    (join, the, SHALLOW),
    (lean, hub, TREE),
]:
    connect(parent, child, color)
mark(113, 157, "2", TREE)
rule(185)

# (c) A cut removes the entire branch, including coffee below a.
header("c", "Score and prune branches", "Steps 1–3", 198)
label("Child score = parent score × edge match rate", 0, 214, 11, MUTED)
a = token(1, 234, "a", "pruned")
coffee = token(59, 234, "coffee", "pruned")
connect(a, coffee, TREE)
token(142, 234, "Hub", "pruned")
token(231, 234, "the", "pruned")
for x, step in [(106, 1), (189, 2), (278, 3)]:
    label(f"✂{step}", x, 234, 11.3, BAD, font=3)
label("Prune a node’s subtree if its path score ≤ τ.", 0, 254, 11, MUTED)
rule(266)

# (d) Above: the retained subtree. Below: the next token predicted at each position.
header("d", "Verify and append tokens", "Steps 3–5", 279)
label("Positions retained after step 3 (new root at left)", 0, 296, 11, MUTED)
positions = [11, 84, 157, 230]
chain = [
    token(x, 317, word, "root" if i == 0 else "live")
    for i, (x, word) in enumerate(zip(positions, ["to", "join", "Lean", "Club"], strict=True))
]
for parent, child, color in zip(chain[:-1], chain[1:], [TREE, TREE, SHALLOW], strict=True):
    connect(parent, child, color)
mark(55, 309, "3", OK)
for left, right, remaining in [
    (positions[0], positions[2] + 44, "One more loop each"),
    (positions[3], positions[3] + 44, "Two more loops"),
]:
    line([(left, 330), (left, 334), (right, 334), (right, 330)])["strokeWidth"] = 0.75
    label(remaining, (left + right) / 2, 343, 10.8, MUTED, "center")
for x in positions:
    line([(x + 22, 351), (x + 22, 363)], MUTED, head=True)
for x, word, step in zip(
    positions, ["join", "Lean", "Club", r"\n"], ["4", "4", "4", "5"], strict=True
):
    token(x, 373, word)
    mark(x + 44, 365, step, OK)
mark(8, 395, "k", OK)
label("Step when this token is appended to the output", 20, 395, 10.8, MUTED)
label("Start a new tree if no child matches the greedy token.", 0, 412, 10.3, MUTED)

scene = {**original, "elements": elements, "files": {}}
body = json.dumps(scene, ensure_ascii=False, indent=1)
texts = "\n".join(f"{e['text']} ^{e['id']}\n" for e in elements if e["type"] == "text")
front = original_md.split("## Text Elements\n", 1)[0]
(OUT / "loopfill-overview.excalidraw.md").write_text(
    front + "## Text Elements\n" + texts + "\n%%\n## Drawing\n```json\n" + body + "\n```\n%%\n"
)
(OUT / "loopfill-overview.excalidraw").write_text(body + "\n")
