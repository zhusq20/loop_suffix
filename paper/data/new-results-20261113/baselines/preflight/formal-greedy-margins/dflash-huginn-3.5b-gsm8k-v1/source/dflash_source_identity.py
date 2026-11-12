"""Strict source checks allowing only reviewed CUDA-startup entry changes."""
import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OLD_ENTRY = 'b6e27deba90ac3b0a52d53f17556c2022b3419f206e399573829cd2c661edaac'
CONTEXT_ENTRY = 'b0c8de5a93fd1534c47cc52f1a4952aa69108cb4a4096ea57208fb11552cb14d'
NEW_ENTRY = '5c892070cc4a804866ec7cdc46b1db31935d4decebfe33c81f2a3caae692168a'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_main(path):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != 'main']
    return ast.dump(tree, include_attributes=False)


def validate_evaluation_sources(folder):
    manifest = json.loads((folder/'source-manifest.json').read_text())
    for relative, expected in manifest.items():
        archived, current = folder/'source'/relative, BASE/relative
        assert sha(archived) == expected, f'Archived evaluation source changed: {relative}'
        actual = sha(current)
        if actual == expected:
            continue
        assert relative == 'evaluate_dflash.py' and expected in {OLD_ENTRY, CONTEXT_ENTRY} and actual == NEW_ENTRY, \
            f'Live DFlash source changed: {relative}'
        assert without_main(archived) == without_main(current), 'Change extends beyond CUDA startup'
    metadata = json.loads((folder/'metadata.json').read_text())
    assert all(manifest[relative] == expected for relative, expected in metadata['source_sha256'].items())
    return manifest
