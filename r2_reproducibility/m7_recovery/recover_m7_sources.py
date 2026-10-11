"""Recover small, pinned M7 inputs without altering the scientific source contract.

The CLI reads only the inventory shipped beside this file. Library callers may
provide a controlled fixture manifest. Recovery always requires a fresh external
directory; historical payloads are used only in explicit --local-dir mode.

Supported deployment layouts are a helper at the package root, or the exact
<package root>/m7_recovery/recover_m7_sources.py layout. In the latter layout the
whole parent package is protected, including siblings of m7_recovery. Deployments
must preserve that directory name rather than introduce arbitrary nesting.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import urllib.parse
import urllib.request


MAX_BYTES = 1_048_576
DEFAULT_INVENTORY = Path(__file__).with_name("m7_source_inventory.json")
ALLOWED_HOSTS = frozenset({"raw.githubusercontent.com", "www.ebi.ac.uk"})
FAMILIES = frozenset({"microbiomehd", "ena"})


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("SHA256 must be exactly 64 hexadecimal characters")
    return value.lower()


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Unsafe relative file path")
    if any(ord(char) < 32 for char in value):
        raise ValueError("Control character in file path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} or part.endswith((" ", ".")) for part in parts):
        raise ValueError("Unsafe path component")
    # Windows device names and metacharacters are unsafe even on other systems.
    for part in parts:
        if any(char in part for char in '<>"|?*'):
            raise ValueError("Unsafe file name")
        if re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", part.split(".")[0], re.I):
            raise ValueError("Reserved file name")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError("Absolute output path is forbidden")
    return path


def _url(value):
    if not isinstance(value, str) or any(char.isspace() or ord(char) < 32 for char in value) or "\\" in value:
        raise ValueError("Unsafe URL")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError("Only HTTPS URLs on the exact approved hosts are allowed")
    if parsed.username is not None or parsed.password is not None or parsed.port not in {None, 443}:
        raise ValueError("Credentials and nondefault ports are forbidden")
    if parsed.fragment or not parsed.path:
        raise ValueError("URL must have a path and no fragment")
    return parsed


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # urllib calls this before opening the redirect destination.
        _url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_url(url, timeout):
    _url(url)
    # No environment proxies, cookie jar, password manager, or auth handlers.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirect())
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    return opener.open(request, timeout=timeout)


def _overlaps(first, second):
    return first == second or first in second.parents or second in first.parents


def _package_root():
    """Select the root of the documented standalone or m7_recovery layout."""
    module_directory = Path(__file__).resolve().parent
    if module_directory.name.casefold() == "m7_recovery":
        return module_directory.parent
    return module_directory


def _read_small(path):
    with path.open("rb") as stream:
        content = stream.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError("Manifest/provenance exceeds the 1 MiB cap")
    return content


def _prepare(root, inventory_path, family, local_dir):
    if family not in FAMILIES | {"all"}:
        raise ValueError("Unknown selected family")
    root_original = Path(root)
    root = root_original.resolve()
    inventory_path = Path(inventory_path).resolve(strict=True)
    if not inventory_path.is_file():
        raise ValueError("Inventory must be a regular file")
    local_dir = Path(local_dir).resolve(strict=True) if local_dir is not None else None
    if local_dir is not None and not local_dir.is_dir():
        raise ValueError("Local source must be a directory")
    input_trees = [_package_root(), inventory_path.parent]
    if local_dir is not None:
        input_trees.append(local_dir)
    if any(_overlaps(root, tree) for tree in input_trees):
        raise ValueError("Destination must be outside package, inventory, provenance, and local source trees")
    if root_original.exists() or root_original.is_symlink() or root.exists():
        raise ValueError("Destination already exists; use a fresh external root")
    manifest = json.loads(_read_small(inventory_path))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("sources"), list):
        raise ValueError("Inventory must contain a sources array")
    if any(not isinstance(entry, dict) or entry.get("family") not in FAMILIES for entry in manifest["sources"]):
        raise ValueError("Invalid source entry or family")
    sizes = [entry.get("bytes") for entry in manifest["sources"]]
    if any(type(size) is not int or not 0 < size <= MAX_BYTES for size in sizes):
        raise ValueError("Expected sizes must be positive integers no larger than 1 MiB")
    if sum(sizes) > MAX_BYTES:
        raise ValueError("Manifest source size total exceeds 1 MiB")
    selected = [dict(entry) for entry in manifest["sources"] if family == "all" or entry["family"] == family]
    if not selected:
        raise ValueError("No sources selected")
    occupied = {"recovery_receipt.json"}
    keys = set()
    total = 0
    for source in selected:
        key = source.get("key")
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError("Source keys must be nonempty and distinct")
        keys.add(key)
        relative = _relative(source.get("file"))
        if str(relative).casefold().endswith((".partial", ".provenance.json")):
            raise ValueError("Source filename collides with a reserved recovery suffix")
        size = source.get("bytes")
        if type(size) is not int or not 0 < size <= MAX_BYTES:
            raise ValueError("Expected size must be a positive integer no larger than 1 MiB")
        total += size
        if total > MAX_BYTES:
            raise ValueError("Selected source size total exceeds 1 MiB")
        source["sha256"] = _sha(source.get("sha256"))
        parsed = _url(source.get("url"))
        if source["family"] == "microbiomehd":
            if parsed.hostname != "raw.githubusercontent.com" or re.fullmatch(r"/[^/]+/[^/]+/[0-9a-fA-F]{40}/.+", parsed.path) is None:
                raise ValueError("MicrobiomeHD requires a commit-pinned raw GitHub URL")
        output_names = [str(relative), str(relative) + ".partial"]
        historical = source.get("historical_provenance")
        if historical is not None:
            if not isinstance(historical, dict):
                raise ValueError("Invalid historical provenance contract")
            provenance_relative = _relative(historical.get("file"))
            if provenance_relative.parts[0] != "provenance":
                raise ValueError("Historical receipt must be in bundled provenance/")
            provenance = inventory_path.parent.joinpath(*provenance_relative.parts).resolve(strict=True)
            if not provenance.is_relative_to(inventory_path.parent / "provenance") or not provenance.is_file():
                raise ValueError("Historical receipt escapes bundled provenance tree")
            content = _read_small(provenance)
            if hashlib.sha256(content).hexdigest() != _sha(historical.get("sha256")):
                raise ValueError("Historical receipt SHA256 mismatch")
            original = json.loads(content)
            if not isinstance(original, dict) or original.get("url") != source["url"] or type(original.get("bytes")) is not int or original["bytes"] != size or original.get("sha256") != source["sha256"]:
                raise ValueError("Historical receipt does not bind expected URL, size and SHA256")
            source["_historical_bytes"] = content
            output_names.append(str(relative) + ".provenance.json")
        # Detect collisions, including case-insensitive names and ancestor paths.
        for name in output_names:
            folded = name.casefold()
            if any(folded == old or folded.startswith(old + "/") or old.startswith(folded + "/") for old in occupied):
                raise ValueError("Duplicate or overlapping output paths")
            occupied.add(folded)
        source["_relative"] = relative
    return root, selected, local_dir


def _transfer(stream, partial, expected_size, expected_sha):
    digest = hashlib.sha256()
    count = 0
    with partial.open("xb") as output:
        while True:
            # One extra byte detects an oversized source, but is never written.
            block = stream.read(min(65536, expected_size - count + 1))
            if not block:
                break
            if len(block) + count > expected_size or len(block) + count > MAX_BYTES:
                raise ValueError("Response exceeds expected size; source contract unchanged")
            output.write(block)
            digest.update(block)
            count += len(block)
    if count != expected_size:
        raise ValueError(f"Source size mismatch: expected {expected_size}, observed {count}")
    sha = digest.hexdigest()
    if sha != expected_sha:
        raise ValueError("Source SHA256 mismatch; source contract unchanged")
    return count, sha


def recover(root, inventory_path, family, local_dir=None):
    """Return a PASS/FAILED receipt; invalid inputs cause no writes or requests.

    On an acquisition failure, preserve any .partial file and stop the command.
    No existing destination is reused, replaced or deleted. The only network
    boundary is _open_url, allowing controlled transport injection in tests.
    """
    receipt = {"status": "FAILED", "started_utc": _utc(), "selected_family": family,
               "retrieval_mode": "local_verified" if local_dir is not None else "remote_verified",
               "completed_files": [], "failures": []}
    try:
        destination, selected, local = _prepare(root, inventory_path, family, local_dir)
    except Exception as error:
        receipt["failures"].append({"stage": "preflight", "error": str(error)})
        receipt["finished_utc"] = _utc()
        return receipt
    created = False
    active = None
    try:
        destination.mkdir(parents=True, exist_ok=False)
        created = True
        for source in selected:
            active = source
            relative = source["_relative"]
            final = destination.joinpath(*relative.parts)
            partial = final.with_name(final.name + ".partial")
            final.parent.mkdir(parents=True, exist_ok=True)
            if local is not None:
                original = local.joinpath(*relative.parts)
                resolved = original.resolve(strict=True)
                if original.is_symlink() or not resolved.is_relative_to(local) or not resolved.is_file():
                    raise ValueError("Local source must be an exact-named regular file within the explicit source directory")
                stream = resolved.open("rb")
            else:
                stream = _open_url(source["url"], timeout=20)
            with closing(stream):
                size, sha = _transfer(stream, partial, source["bytes"], source["sha256"])
            # Fresh destination and exclusive .partial prevent implicit reuse.
            if final.exists() or final.is_symlink():
                raise ValueError("Final source unexpectedly exists")
            partial.rename(final)
            current = {"key": source["key"], "family": source["family"], "file": source["file"],
                       "url": source["url"], "bytes": size, "sha256": sha,
                       "retrieved_utc": _utc(), "retrieval_mode": receipt["retrieval_mode"]}
            if local is not None:
                current["local_source"] = str(local.joinpath(*relative.parts))
            if "_historical_bytes" in source:
                historical_file = source["file"] + ".provenance.json"
                with final.with_name(final.name + ".provenance.json").open("xb") as output:
                    output.write(source["_historical_bytes"])
                current["historical_provenance"] = {"file": historical_file, "kind": "historical",
                                                    "sha256": source["historical_provenance"]["sha256"]}
            receipt["completed_files"].append(current)
        receipt["status"] = "PASS"
    except Exception as error:
        receipt["failures"].append({"stage": "recovery", "file": active["file"] if active else None,
                                    "error": str(error)})
    receipt["finished_utc"] = _utc()
    if created:
        try:
            with (destination / "recovery_receipt.json").open("x", encoding="utf-8") as output:
                json.dump(receipt, output, indent=2)
                output.write("\n")
        except Exception as error:
            receipt["status"] = "FAILED"
            receipt["failures"].append({"stage": "receipt", "error": str(error)})
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Fresh directory outside all input/package trees")
    parser.add_argument("--family", required=True, choices=["microbiomehd", "ena", "all"])
    parser.add_argument("--local-dir", type=Path, help="Explicit historical/local payload source; never counts as remote retrieval")
    args = parser.parse_args(argv)
    receipt = recover(args.root, DEFAULT_INVENTORY, args.family, args.local_dir)
    print(json.dumps(receipt, indent=2), file=sys.stdout if receipt["status"] == "PASS" else sys.stderr)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
