"""Portable configuration, exact TSV parsing and protected fresh destinations."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .evaluation import validate_predictions

REPOSITORY = Path(__file__).resolve().parents[1]
PATH_KEYS = {"source_root", "training_manifest", "output_root", "predictions",
             "metadata", "state", "model_bundle", "baseline_predictions",
             "historical_directions"}
LIST_PATH_KEYS = {"evaluation_dirs", "state_dirs", "blind_dirs", "model_dirs"}


def load_config(path):
    path = Path(path).resolve(strict=True)
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be an object")
    for key in PATH_KEYS & config.keys():
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"Nonempty path required: {key}")
        config[key] = (path.parent / config[key]).resolve()
    for key in LIST_PATH_KEYS & config.keys():
        if not isinstance(config[key], list) or not all(isinstance(p, str) and p for p in config[key]):
            raise ValueError(f"Path list required: {key}")
        config[key] = [(path.parent / p).resolve() for p in config[key]]
    config["_config_path"] = path
    return config


def overlap(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    return a == b or a in b.parents or b in a.parents


def validate_output(output, sources):
    output = Path(output).resolve()
    if output == output.anchor or output == Path(output.anchor):
        raise ValueError("Filesystem root is not an output")
    for source in [REPOSITORY, *sources]:
        if overlap(output, source):
            raise ValueError(f"Output overlaps protected input/code: {source}")
    return output


def protected_paths(config):
    paths = [value for key, value in config.items() if key in PATH_KEYS and key != "output_root"]
    for key in LIST_PATH_KEYS & config.keys():
        paths.extend(config[key])
    # Protect entire parent input trees of explicitly supplied artifacts.
    return [p if Path(p).is_dir() else Path(p).parent for p in paths]


def fresh(directory, sources):
    directory = validate_output(directory, sources)
    if directory.exists():
        raise FileExistsError(f"Fresh destination required: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def read_tsv(path, **kwargs):
    path = Path(path)
    opener = __import__("gzip").open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, delimiter="\t"), [])
    if not header or len(header) != len(set(header)):
        raise ValueError(f"Empty or duplicate TSV columns: {path}")
    return pd.read_csv(path, sep="\t", float_precision="round_trip", **kwargs)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_table(path, frame):
    frame.to_csv(path, sep="\t", index=False, float_format="%.17g")


def seal(directory, action, inputs, details=None, expected_inputs=None):
    directory = Path(directory)
    current_inputs = {str(Path(p).resolve()): sha(p) for p in inputs}
    if expected_inputs is not None:
        for name, digest in current_inputs.items():
            if name not in expected_inputs or digest != expected_inputs[name]:
                raise ValueError(f"Protected input changed or was not frozen: {name}")
    outputs = {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file()}
    receipt = {"status": "PASS", "action": action,
               "completed_utc": datetime.now(timezone.utc).isoformat(),
               "inputs": current_inputs,
               "outputs": outputs, **(details or {})}
    save_json(directory / "receipt.json", receipt)
    return receipt


def check_receipt(directory):
    directory = Path(directory).resolve()
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS" or not receipt.get("outputs"):
        raise ValueError("Incomplete/nonpassing artifact receipt")
    for name, digest in receipt["outputs"].items():
        path = (directory / name).resolve()
        if path.parent != directory or sha(path) != digest:
            raise ValueError(f"Changed or unsafe receipted output: {name}")
    return receipt


def point_replay(predictions, metadata):
    aligned = validate_predictions(predictions, metadata)
    rows = []
    for (study, key), part in aligned.groupby(["study_name", "model_key"], sort=True):
        y, p = part.y.to_numpy(), part.probability_crc.to_numpy()
        if set(y) != {0, 1}:
            raise ValueError("Both classes required for point metrics")
        panel, classifier = key.split("::")
        rows.append({"study_name": study, "model_key": key, "panel": panel,
                     "classifier": classifier, "n": len(part), "k": int(part.k.iloc[0]),
                     "auroc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p))})
    return pd.DataFrame(rows)
