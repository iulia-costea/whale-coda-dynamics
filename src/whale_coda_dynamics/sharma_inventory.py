"""Sharma et al.'s rhythm templates and tempo classes as plain JSON (PLAN [R2-1], [R2-2]; DEC-22).

`mean_codas.p` (the 18 rhythm templates) and `tempos-dict.p` (the durations in each tempo class)
hold NumPy objects, so unpickling them needs NumPy. This module runs once; its output,
`data/interim/sharma_inventory.json`, is committed so that `preprocessing/inventories.py` stays
standard-library only. `tests/test_inventories.py` re-extracts it when NumPy is available.
"""
import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

from .preprocessing.io_utils import read_json, write_text


def _finite(value, what):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value in {what}")
    return value


def templates_from(mean_codas):
    """Templates in their original order (index = rhythm label), as lists of floats."""
    if not isinstance(mean_codas, list) or not mean_codas:
        raise ValueError("mean_codas.p must hold a nonempty list of templates")
    templates = []
    for k, template in enumerate(mean_codas):
        if getattr(template, "ndim", 1) != 1:
            raise ValueError(f"Template {k} is not one-dimensional")
        positions = [_finite(v, f"template {k}") for v in template]
        if not 3 <= len(positions) <= 10:
            raise ValueError(f"Template {k} has {len(positions)} clicks; expected 3-10")
        templates.append(positions)
    return templates


def tempo_classes_from(tempos):
    """Count, shortest and longest duration per class; boundaries midway between adjacent classes.

    Empty classes are recorded and left out. The non-empty classes must be keys 0..k-1, in
    increasing order of duration and without overlap, or the boundaries would be ambiguous.
    """
    if not isinstance(tempos, dict) or any(type(key) is not int for key in tempos):
        raise ValueError("tempos-dict.p must map integer class keys to duration lists")
    classes = []
    for key in sorted(tempos):
        durations = [_finite(v, f"tempo class {key}") for v in tempos[key]]
        classes.append({"class": key, "count": len(durations),
                        "min_duration_s": min(durations) if durations else None,
                        "max_duration_s": max(durations) if durations else None})
    filled = [c for c in classes if c["count"]]
    if [c["class"] for c in filled] != list(range(len(filled))) or len(filled) < 2:
        raise ValueError("Non-empty tempo classes must be keys 0..k-1 (k >= 2)")
    boundaries = []
    for lower, upper in zip(filled, filled[1:]):
        if not lower["max_duration_s"] < upper["min_duration_s"]:
            raise ValueError(f"Tempo classes {lower['class']} and {upper['class']} overlap")
        boundaries.append((lower["max_duration_s"] + upper["min_duration_s"]) / 2)
    return classes, boundaries


def extract(root, config):
    root = Path(root)
    paths = {name: root / config[name] for name in ("mean_codas", "tempos_dict")}
    # Binary pickles, not text: io_utils covers text I/O only.
    templates = templates_from(pickle.loads(paths["mean_codas"].read_bytes()))
    classes, boundaries = tempo_classes_from(pickle.loads(paths["tempos_dict"].read_bytes()))
    return {
        "source": {"repository": config["source_repository"], "commit": config["source_commit"],
                   "files": [config["mean_codas"], config["tempos_dict"]]},
        "input_sha256": {config[name]: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "rhythm_templates": templates,
        "template_click_counts": [len(t) for t in templates],
        "tempo_classes": classes,
        "empty_tempo_classes": [c["class"] for c in classes if not c["count"]],
        "tempo_boundaries_s": boundaries,
        "boundary_rule": "midpoint between the longest duration of each non-empty class and the shortest of the next",
    }


def serialize(result):
    return json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def run(config_path):
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    result = extract(root, config)
    output = root / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_text(output, serialize(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/sharma_inventory.json")
    result = run(parser.parse_args().config)
    print(f"Extracted {len(result['rhythm_templates'])} rhythm templates and "
          f"{len(result['tempo_boundaries_s'])} tempo boundaries {result['tempo_boundaries_s']}.")


if __name__ == "__main__":
    main()
