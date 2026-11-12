"""Small immutable source bundles stored beside each experiment journal."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def freeze_sources(out, paths):
    manifest = {}
    for path in paths:
        path = Path(path).resolve()
        name = str(path.relative_to(BASE))
        data = path.read_bytes()
        dest = Path(out)/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.read_bytes() != data:
                raise RuntimeError(f'Resuming with changed source is not allowed: {name}')
        else:
            dest.write_bytes(data)
        manifest[name] = hashlib.sha256(data).hexdigest()
    (Path(out)/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
