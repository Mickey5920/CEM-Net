"""Read shared source snapshots and export an exact executed recipe."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(path):
    value = json.loads(Path(path).read_text(encoding="utf8"))
    if isinstance(value, dict) and value.get("format") == "cem-net-source-links-v1":
        out = {}
        for name, rel in value["files"].items():
            p = (ROOT / rel).resolve()
            if not p.is_relative_to(ROOT) or not p.is_file():
                raise ValueError(f"Invalid source reference: {rel}")
            out[name] = base64.b64encode(p.read_bytes()).decode("ascii")
        return out
    return value

def snapshots():
    return sorted(p.parent.relative_to(ROOT / "source").as_posix()
                  for p in (ROOT / "source").rglob("source_bundle.json"))

def export(name, output):
    if name not in snapshots():
        raise ValueError("Choose a snapshot listed by `python source_loader.py list`.")
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(ROOT) or ROOT.is_relative_to(output):
        raise ValueError("Use a new output directory outside this artifact.")
    source = ROOT / "source" / name
    expected = read(source / "source_manifest.json")
    encoded = read(source / "source_bundle.json")
    if expected.keys() != encoded.keys():
        raise ValueError("Source manifest does not match snapshot.")
    output.mkdir(parents=True)
    for rel, value in encoded.items():
        dest = (output / rel).resolve()
        if not dest.is_relative_to(output):
            raise ValueError(f"Invalid output path: {rel}")
        data = base64.b64decode(value)
        if hashlib.sha256(data).hexdigest() != expected[rel]:
            raise ValueError(f"Source checksum mismatch: {rel}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    (output / "source_manifest.json").write_bytes((source / "source_manifest.json").read_bytes())
    return {"snapshot": name, "files": len(expected), "output": str(output), "training_executed": False}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("list", "export"))
    p.add_argument("snapshot", nargs="?")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.mode == "list":
        print("\n".join(snapshots()))
    else:
        if not a.snapshot or a.output is None:
            p.error("export requires a snapshot and --output")
        print(json.dumps(export(a.snapshot, a.output), indent=2))
