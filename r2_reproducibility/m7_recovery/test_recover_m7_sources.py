"""Controlled fixtures: no live network and no historical payload reuse."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.request
import urllib.response
from email.message import Message

MODULE = Path(__file__).with_name("recover_m7_sources.py")
SPEC = importlib.util.spec_from_file_location("recovery", MODULE)
recovery = importlib.util.module_from_spec(SPEC) if MODULE.exists() else None
if recovery is not None:
    SPEC.loader.exec_module(recovery)

ABC_SHA = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
URL = "https://raw.githubusercontent.com/owner/repo/" + "a" * 40 + "/abc.txt"


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(recovery, "Recovery helper has not been implemented")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.contract = self.base / "contract"
        self.contract.mkdir()
        self.local = self.base / "local"
        self.local.mkdir()
        (self.local / "abc.txt").write_bytes(b"abc")
        self.root = self.base / "recovered"
        self.inventory = self.contract / "m7_source_inventory.json"
        self.entry = dict(key="abc", family="microbiomehd", file="abc.txt",
                          url=URL, bytes=3, sha256=ABC_SHA)
        self.write_manifest([self.entry])

    def write_manifest(self, sources):
        self.inventory.write_text(json.dumps({"sources": sources}), encoding="utf-8")

    def run_local(self):
        return recovery.recover(self.root, self.inventory, "all", self.local)

    def test_verified_local_has_literal_hash_and_fresh_receipt(self):
        receipt = self.run_local()
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual((self.root / "abc.txt").read_bytes(), b"abc")
        self.assertEqual(receipt["completed_files"][0]["sha256"], ABC_SHA)
        self.assertEqual(receipt["completed_files"][0]["retrieval_mode"], "local_verified")
        self.assertIn("retrieved_utc", receipt["completed_files"][0])
        self.assertEqual(json.loads((self.root / "recovery_receipt.json").read_text())["status"], "PASS")

    def test_tampered_local_fails_with_partial_and_error_receipt(self):
        (self.local / "abc.txt").write_bytes(b"abd")
        result = self.run_local()
        self.assertEqual(result["status"], "FAILED")
        self.assertFalse((self.root / "abc.txt").exists())
        self.assertTrue((self.root / "abc.txt.partial").exists())
        self.assertTrue(json.loads((self.root / "recovery_receipt.json").read_text())["failures"])

    def test_wrong_size_fails_and_cli_exits_nonzero(self):
        (self.local / "abc.txt").write_bytes(b"ab")
        self.assertEqual(self.run_local()["status"], "FAILED")
        with patch.object(recovery, "DEFAULT_INVENTORY", self.inventory), patch("sys.stderr", new_callable=io.StringIO):
            self.assertNotEqual(recovery.main(["--root", str(self.base / "cli"), "--family", "all", "--local-dir", str(self.local)]), 0)

    def test_invalid_contracts_have_no_output_or_network_side_effects(self):
        variants = [
            {"url": "http://www.ebi.ac.uk/abc"},
            {"url": "https://evil.example/abc"},
            {"url": "https://user:secret@www.ebi.ac.uk/abc"},
            {"url": "https://www.ebi.ac.uk:444/abc"},
            {"url": "https://raw.githubusercontent.com/owner/repo/main/abc"},
            {"file": "../abc"}, {"file": "/abc"}, {"file": "C:/abc"},
            {"file": "dir\\abc"}, {"file": "./abc"},
            {"file": "recovery_receipt.json"}, {"file": "abc.txt.partial"},
            {"bytes": 0}, {"bytes": True}, {"bytes": 1048577},
            {"sha256": "broken"}, {"family": "unknown"},
        ]
        for changes in variants:
            with self.subTest(changes=changes):
                self.write_manifest([dict(self.entry, **changes)])
                with patch.object(recovery, "_open_url", side_effect=AssertionError("network forbidden")):
                    result = recovery.recover(self.root, self.inventory, "all")
                self.assertEqual(result["status"], "FAILED")
                self.assertFalse(self.root.exists())

    def test_duplicate_collision_and_total_size_rejected_before_writes(self):
        pairs = [self.entry.copy(), dict(self.entry, key="second", file="ABC.TXT")]
        self.write_manifest(pairs)
        self.assertEqual(self.run_local()["status"], "FAILED")
        self.assertFalse(self.root.exists())
        self.write_manifest([dict(self.entry, bytes=700000), dict(self.entry, key="second", file="b", bytes=700000)])
        self.assertEqual(self.run_local()["status"], "FAILED")
        self.assertFalse(self.root.exists())

    def test_existing_destination_preserved(self):
        self.root.mkdir()
        marker = self.root / "keep.txt"
        marker.write_bytes(b"keep")
        self.assertEqual(self.run_local()["status"], "FAILED")
        self.assertEqual(marker.read_bytes(), b"keep")
        self.assertEqual(list(self.root.iterdir()), [marker])

    def test_manifest_total_cap_applies_even_with_family_filter(self):
        self.write_manifest([dict(self.entry, bytes=600000), dict(self.entry, key="ena", family="ena", file="other", bytes=600000, url="https://www.ebi.ac.uk/ena/portal/api/search")])
        result = recovery.recover(self.root, self.inventory, "microbiomehd", self.local)
        self.assertEqual(result["status"], "FAILED")
        self.assertFalse(self.root.exists())

    def test_recovery_suffix_reserved_case_insensitively(self):
        self.write_manifest([dict(self.entry, file="ABC.PARTIAL")])
        result = self.run_local()
        self.assertEqual(result["status"], "FAILED")
        self.assertFalse(self.root.exists())

    def test_output_must_not_overlap_any_input_tree(self):
        for root in [self.contract / "child", self.local / "child", self.base / "ancestor", MODULE.parent / "child"]:
            if root.name == "ancestor":
                # An absent ancestor of the manifest cannot occur; existing ancestors are refused too.
                root = self.base
            with self.subTest(root=root):
                result = recovery.recover(root, self.inventory, "all", self.local)
                self.assertEqual(result["status"], "FAILED")
        self.assertFalse((self.contract / "child").exists())
        self.assertFalse((self.local / "child").exists())

    def test_nested_package_layout_protects_sibling_and_allows_external_root(self):
        package = self.base / "fixture_package"
        nested = package / "m7_recovery"
        nested.mkdir(parents=True)
        module_path = nested / "recover_m7_sources.py"
        shutil.copyfile(MODULE, module_path)
        manifest = nested / "m7_source_inventory.json"
        manifest.write_bytes(self.inventory.read_bytes())
        spec = importlib.util.spec_from_file_location("nested_recovery", module_path)
        integrated = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(integrated)
        inside = package / "sibling_output"
        with patch.object(integrated, "_open_url", side_effect=AssertionError("network forbidden")):
            blocked = integrated.recover(inside, manifest, "all")
        self.assertEqual(blocked["status"], "FAILED")
        self.assertFalse(inside.exists())
        self.assertTrue(any("outside package" in error["error"] for error in blocked["failures"]))
        allowed = integrated.recover(self.root, manifest, "all", self.local)
        self.assertEqual(allowed["status"], "PASS")
        self.assertEqual((self.root / "abc.txt").read_bytes(), b"abc")

    def test_verified_network_uses_bounded_read_and_no_retry(self):
        with patch.object(recovery, "_open_url", return_value=io.BytesIO(b"abc")) as opener:
            result = recovery.recover(self.root, self.inventory, "microbiomehd")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["completed_files"][0]["retrieval_mode"], "remote_verified")
        opener.assert_called_once_with(URL, timeout=20)

    def test_env_proxy_does_not_route_requests_or_add_credentials(self):
        headers = Message()
        response = urllib.response.addinfourl(io.BytesIO(b"abc"), headers, URL, 200)
        response.msg = "OK"
        with patch.dict("os.environ", {"https_proxy": "http://user:secret@proxy.example:1234", "HTTPS_PROXY": "http://user:secret@proxy.example:1234"}):
            with patch.object(urllib.request.HTTPSHandler, "https_open", return_value=response) as transport:
                with recovery._open_url(URL, timeout=20) as opened:
                    self.assertEqual(opened.read(), b"abc")
        req = transport.call_args.args[0]
        self.assertEqual(req.host, "raw.githubusercontent.com")
        self.assertFalse(req.has_header("Authorization"))
        self.assertFalse(req.has_header("Proxy-Authorization"))
        self.assertFalse(req.has_header("Cookie"))

    def test_failed_second_file_keeps_first_completed_and_marks_failed(self):
        self.write_manifest([self.entry, dict(self.entry, key="second", file="second.txt")])
        (self.local / "second.txt").write_bytes(b"wrong")
        result = self.run_local()
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(len(result["completed_files"]), 1)
        self.assertEqual((self.root / "abc.txt").read_bytes(), b"abc")
        self.assertFalse((self.root / "second.txt").exists())
        self.assertTrue((self.root / "second.txt.partial").exists())

    def test_network_oversize_and_error_never_pass_or_retry(self):
        for payload in [b"abcd", OSError("fixture timeout")]:
            with self.subTest(payload=payload):
                root = self.base / ("overflow" if isinstance(payload, bytes) else "timeout")
                kwargs = {"return_value": io.BytesIO(payload)} if isinstance(payload, bytes) else {"side_effect": payload}
                with patch.object(recovery, "_open_url", **kwargs) as opener:
                    result = recovery.recover(root, self.inventory, "all")
                self.assertEqual(result["status"], "FAILED")
                self.assertFalse((root / "abc.txt").exists())
                self.assertEqual(opener.call_count, 1)

    def test_redirect_rejected_before_following_request(self):
        headers = Message()
        headers["Location"] = "https://evil.example/payload"
        response = urllib.response.addinfourl(io.BytesIO(b""), headers, URL, 302)
        response.msg = "Found"
        with patch.object(urllib.request.HTTPSHandler, "https_open", return_value=response) as transport:
            with self.assertRaises(ValueError):
                recovery._open_url(URL, timeout=20)
        self.assertEqual(transport.call_count, 1)

    def test_historical_receipt_exact_bytes_distinct_from_current(self):
        historical = b'{"url": "' + URL.encode() + b'", "bytes":3,"sha256":"' + ABC_SHA.encode() + b'","retrieved_utc":"2020-01-01T00:00:00Z"}\n'
        provenance = self.contract / "provenance"
        provenance.mkdir()
        (provenance / "old.json").write_bytes(historical)
        self.entry["historical_provenance"] = dict(file="provenance/old.json", sha256=hashlib.sha256(historical).hexdigest())
        self.write_manifest([self.entry])
        result = self.run_local()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual((self.root / "abc.txt.provenance.json").read_bytes(), historical)
        current = result["completed_files"][0]
        self.assertEqual(current["historical_provenance"]["kind"], "historical")
        self.assertNotEqual(current["retrieved_utc"], "2020-01-01T00:00:00Z")

    def test_history_tamper_or_binding_mismatch_fails_before_recovery(self):
        provenance = self.contract / "provenance"
        provenance.mkdir()
        for wrong_binding in [False, True]:
            historical = json.dumps(dict(url=URL, bytes=4 if wrong_binding else 3, sha256=ABC_SHA)).encode()
            (provenance / "old.json").write_bytes(historical)
            self.entry["historical_provenance"] = dict(file="provenance/old.json", sha256=hashlib.sha256(historical).hexdigest() if wrong_binding else "0" * 64)
            self.write_manifest([self.entry])
            self.assertEqual(self.run_local()["status"], "FAILED")
            self.assertFalse(self.root.exists())

    def test_family_selection_does_not_recover_other_sources(self):
        self.write_manifest([self.entry, dict(self.entry, key="ena", family="ena", file="other", url="https://www.ebi.ac.uk/ena/portal/api/search")])
        result = recovery.recover(self.root, self.inventory, "microbiomehd", self.local)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["completed_files"]), 1)
        self.assertFalse((self.root / "other").exists())

    def test_cli_has_no_inventory_or_host_override(self):
        result = subprocess.run([sys.executable, str(MODULE), "--root", str(self.root), "--family", "all", "--inventory", str(self.inventory)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
