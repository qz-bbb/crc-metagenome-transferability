#!/usr/bin/env python
"""Exact-size ARP/DCP refits matched to each fold's observed ICP size.

This bounded add-on does not rerun the nested panel-selection analysis.  It
reconstructs the train-only preprocessing used by ``run_nested_loso.py``,
reads each saved training-fold association table, and uses the saved nested
ICP membership to define the target feature count.  ARP and DCP are then
truncated to exactly that count before fitting the locked Elastic Net and
Random Forest models.  The held-out study is never used for panel selection,
size selection, or model fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path('UNCONFIGURED_OUTPUT')
NESTED = HERE.parent
ROOT = NESTED.parent.parent.parent.parent
MATRIX = Path('UNCONFIGURED_MATRIX')
META = Path('UNCONFIGURED_METADATA')
SEED = 20261009
BOOT = 2000


def load_nested_module():
    path = NESTED / "run_nested_loso.py"
    spec = importlib.util.spec_from_file_location("nested_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reference script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.BOOT = BOOT
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_long_membership(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, sep="\t", dtype=str)
    required = {"held_out_study", "panel", "panel_size_reported", "panel_rank", "taxon"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"Membership table missing columns: {sorted(missing)}")
    x["panel_size_reported"] = pd.to_numeric(x["panel_size_reported"], errors="raise")
    x["panel_rank"] = pd.to_numeric(x["panel_rank"], errors="raise")
    # The normalized legacy export retained a terminal taxonomy delimiter for
    # many rows, whereas the source matrix/association tables do not.  Remove
    # only that delimiter; do not alter internal ranks or SGB names.
    x["taxon"] = x["taxon"].astype(str).str.rstrip("|")
    return x


def ordered_panel(membership: pd.DataFrame, held: str, panel: str) -> tuple[list[str], int]:
    g = membership[(membership.held_out_study == held) & (membership.panel == panel)].copy()
    if g.empty:
        raise AssertionError(f"No saved membership for {held}/{panel}")
    g = g.sort_values("panel_rank")
    sizes = g.panel_size_reported.unique().tolist()
    if len(sizes) != 1:
        raise AssertionError(f"Inconsistent saved panel size for {held}/{panel}: {sizes}")
    n = int(sizes[0])
    if n != len(g):
        raise AssertionError(f"Saved panel count mismatch for {held}/{panel}: reported {n}, rows {len(g)}")
    return g.taxon.tolist(), n


def association_rank(association: pd.DataFrame, n: int) -> list[str]:
    """Use the saved fold association q ranking, not a new held-out-aware fit."""
    if not set(["taxon", "q"]).issubset(association.columns):
        raise ValueError("Association table lacks taxon/q columns")
    # pandas places NaN q values last; taxon is the saved deterministic tie-break.
    return association.sort_values(["q", "taxon"], na_position="last").head(n).taxon.tolist()


def dcp_rank(association: pd.DataFrame, n: int) -> list[str]:
    """Apply the locked DCP eligibility rule and then use saved q ranking."""
    need = {"taxon", "q", "direction_agreement"}
    if not need.issubset(association.columns):
        raise ValueError("Association table lacks DCP columns")
    eligible = association[(association.q < 0.10) & (association.direction_agreement >= 0.60)]
    eligible = eligible.sort_values(["q", "taxon"], na_position="last")
    if len(eligible) < n:
        raise AssertionError(f"DCP has only {len(eligible)} eligible taxa but target size is {n}")
    return eligible.head(n).taxon.tolist()


def fold_data(mod, matrix: pd.DataFrame, meta: pd.DataFrame, held: str):
    common = [s for s in matrix.columns if s in meta.index]
    matrix = matrix.loc[:, common]
    meta = meta.loc[common]
    keep = meta["study_condition"].isin(["CRC", "control"])
    matrix = matrix.loc[:, keep]
    meta = meta.loc[keep]
    y = (meta.study_condition == "CRC").astype(int).to_numpy()
    studies = meta.study_name.to_numpy()
    taxa = matrix.index.to_numpy()
    arr = matrix.to_numpy().T
    tr = studies != held
    te = ~tr
    xtr0 = arr[tr]
    xte0 = arr[te]
    ytr = y[tr]
    yte = y[te]
    str_ = studies[tr]
    prev = (xtr0 > 0).mean(axis=0)
    ns = np.array([len(set(str_[xtr0[:, j] > 0])) for j in range(arr.shape[1])])
    sel = (prev >= 0.05) & (ns >= 6)
    selected = taxa[sel]
    xtr0 = xtr0[:, sel]
    xte0 = xte0[:, sel]
    positive = xtr0[xtr0 > 0]
    p = float(np.min(positive) / 2) if len(positive) else 1e-12
    xtr = mod.clr_transform(xtr0, p)
    xte = mod.clr_transform(xte0, p)
    return {
        "meta": meta,
        "y": y,
        "studies": studies,
        "taxa": taxa,
        "arr": arr,
        "tr": tr,
        "te": te,
        "xtr0": xtr0,
        "xte0": xte0,
        "xtr": xtr,
        "xte": xte,
        "ytr": ytr,
        "yte": yte,
        "str_": str_,
        "selected": selected,
        "pseudocount": p,
    }


def taxa_indices(selected: np.ndarray, taxa: list[str]) -> list[int]:
    pos = {str(t): i for i, t in enumerate(selected)}
    missing = [t for t in taxa if t not in pos]
    if missing:
        raise AssertionError(f"Panel taxa not in fold-selected feature universe: {missing[:3]}")
    return [pos[t] for t in taxa]


def main() -> None:
    global BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    BOOT = int(args.bootstrap)
    started = time.time()
    OUT = HERE
    OUT.mkdir(parents=True, exist_ok=True)
    mod = load_nested_module()
    mod.BOOT = BOOT
    mod.tie_unit_test()

    matrix = mod.read_matrix(MATRIX)
    meta = pd.read_csv(META, sep="\t").set_index("sample_uid")
    common = [s for s in matrix.columns if s in meta.index]
    matrix = matrix.loc[:, common]
    meta = meta.loc[common]
    keep = meta["study_condition"].isin(["CRC", "control"])
    matrix = matrix.loc[:, keep]
    meta = meta.loc[keep]
    studies_order = list(pd.unique(meta.study_name.to_numpy()))
    if args.smoke:
        studies_order = studies_order[:1]
    membership = parse_long_membership(NESTED / "nested_loso_panel_membership_long.tsv")

    all_metrics: list[dict] = []
    all_predictions: list[dict] = []
    all_membership: list[dict] = []
    errors: list[dict] = []
    fold_audits: list[dict] = []
    panels = ["ARP_nICP", "DCP_nICP", "ICP", "prevalence_matched_random_nICP"]
    models = ["ElasticNet", "RF"]

    for fold_i, held in enumerate(studies_order):
        try:
            d = fold_data(mod, matrix, meta, held)
            assoc_path = NESTED / f"fold_{fold_i:02d}_association.tsv"
            assoc = pd.read_csv(assoc_path, sep="\t")
            if assoc.taxon.tolist() != d["selected"].tolist():
                raise AssertionError(f"fold {fold_i} saved association taxa do not equal reconstructed selected universe")
            # Saved target panels define the actual ICP size and negative-control set.
            icp_taxa, n_icp = ordered_panel(membership, held, "ICP")
            random_taxa, n_random = ordered_panel(membership, held, "prevalence_matched_random_nICP")
            if n_icp != n_random:
                raise AssertionError(f"{held}: random panel size {n_random} != ICP size {n_icp}")
            arp_taxa = association_rank(assoc, n_icp)
            dcp_taxa = dcp_rank(assoc, n_icp)
            panel_taxa = {
                "ARP_nICP": arp_taxa,
                "DCP_nICP": dcp_taxa,
                "ICP": icp_taxa,
                "prevalence_matched_random_nICP": random_taxa,
            }
            # Every evaluated panel is exactly nICP and can be mapped without zero-fill.
            for pn, tax in panel_taxa.items():
                if len(tax) != n_icp or len(set(tax)) != n_icp:
                    raise AssertionError(f"{held}/{pn}: panel size is not exactly nICP={n_icp}")
                idx = taxa_indices(d["selected"], tax)
                for rank, taxon in enumerate(tax, start=1):
                    all_membership.append({
                        "held_out_study": held,
                        "panel": pn,
                        "panel_size": n_icp,
                        "panel_rank": rank,
                        "taxon": taxon,
                        "selected_features": len(d["selected"]),
                    })
                for model in models:
                    pred = mod.fit_predict(d["xtr"][:, idx], d["ytr"], d["xte"][:, idx], "RF" if model == "RF" else "EN")
                    met = mod.metrics(d["yte"], pred, SEED + fold_i)
                    all_metrics.append({
                        "held_out_study": held,
                        "model": model,
                        "panel": pn,
                        "panel_size": n_icp,
                        "n_train": int(d["tr"].sum()),
                        "n_test": int(d["te"].sum()),
                        "n_train_studies": int(len(set(d["str_" ]))),
                        "prevalence_min": 0.05,
                        "study_min": 6,
                        "pseudocount": d["pseudocount"],
                        "nested_preprocessing": True,
                        **met,
                    })
                    for sample, yy, pp in zip(d["meta"].index[d["te"]], d["yte"], pred):
                        all_predictions.append({
                            "held_out_study": held,
                            "model": model,
                            "panel": pn,
                            "sample_uid": sample,
                            "y": int(yy),
                            "pred": float(pp),
                        })
            fold_audits.append({
                "held_out_study": held,
                "fold_index": fold_i,
                "selected_features": int(len(d["selected"])),
                "nICP": int(n_icp),
                "ARP_nICP_n": len(arp_taxa),
                "DCP_nICP_n": len(dcp_taxa),
                "ICP_n": len(icp_taxa),
                "random_nICP_n": len(random_taxa),
                "saved_association_sha256": sha256(assoc_path),
            })
        except Exception as exc:
            errors.append({"held_out_study": held, "fold_index": fold_i, "error": repr(exc)})

    metrics = pd.DataFrame(all_metrics)
    predictions = pd.DataFrame(all_predictions)
    memberships = pd.DataFrame(all_membership)
    audits = pd.DataFrame(fold_audits)
    metrics.to_csv(OUT / "matched_size_metrics.tsv", sep="\t", index=False)
    predictions.to_csv(OUT / "matched_size_predictions.tsv.gz", sep="\t", index=False, compression="gzip")
    memberships.to_csv(OUT / "matched_size_panel_membership.tsv", sep="\t", index=False)
    membership_groups = (
        memberships.groupby(["held_out_study", "panel"], as_index=False, sort=False)
        .agg(
            panel_size=("panel_size", "first"),
            selected_features=("selected_features", "first"),
            taxa=("taxon", lambda s: "||".join(s.tolist())),
        )
    )
    membership_groups.to_csv(OUT / "matched_size_panel_membership_groups.tsv", sep="\t", index=False)
    audits.to_csv(OUT / "matched_size_fold_audit.tsv", sep="\t", index=False)
    pd.DataFrame(errors).to_csv(OUT / "matched_size_errors.tsv", sep="\t", index=False)

    # Compact medians and fold-level paired differences.  Existing ICP and
    # prevalence-matched rows are the nested run's fixed references; the
    # exact-size refits are compared without reusing held-out labels for any
    # selection step.
    if not metrics.empty:
        rows = []
        for (panel, model), g in metrics.groupby(["panel", "model"], sort=True):
            rows.append({
                "panel": panel,
                "model": model,
                "folds": len(g),
                "median_panel_size": float(g.panel_size.median()),
                "median_auroc": float(g.auroc.median()),
                "mean_auroc": float(g.auroc.mean()),
                "min_auroc": float(g.auroc.min()),
                "median_auprc": float(g.auprc.median()),
                "mean_auprc": float(g.auprc.mean()),
                "min_auprc": float(g.auprc.min()),
                "bootstrap_valid_min": int(g.bootstrap_valid.min()),
                "bootstrap_valid_max": int(g.bootstrap_valid.max()),
            })
        pd.DataFrame(rows).to_csv(OUT / "matched_size_summary.tsv", sep="\t", index=False)

    ref_metrics = pd.read_csv(NESTED / "nested_loso_metrics.tsv", sep="\t")
    ref = ref_metrics[ref_metrics.panel.isin(["ICP", "prevalence_matched_random_nICP"])].copy()
    cmp = metrics.merge(ref[["held_out_study", "model", "panel", "auroc", "auprc"]], on=["held_out_study", "model", "panel"], suffixes=("_matched_size", "_reference"), how="left")
    # Add the exact-size panels against both fixed-size references, so the
    # direction of any size effect is explicit in the receipt tables.
    rows = []
    for _, row in metrics[metrics.panel.isin(["ARP_nICP", "DCP_nICP"])].iterrows():
        for ref_panel in ["ICP", "prevalence_matched_random_nICP"]:
            rr = ref_metrics[(ref_metrics.held_out_study == row.held_out_study) & (ref_metrics.model == row.model) & (ref_metrics.panel == ref_panel)]
            if len(rr) != 1:
                raise AssertionError(f"Missing reference metric for {row.held_out_study}/{row.model}/{ref_panel}")
            rr = rr.iloc[0]
            rows.append({
                "held_out_study": row.held_out_study,
                "model": row.model,
                "matched_panel": row.panel,
                "reference_panel": ref_panel,
                "matched_panel_size": int(row.panel_size),
                "reference_panel_size": int(rr.panel_size),
                "auroc_difference": float(row.auroc - rr.auroc),
                "auprc_difference": float(row.auprc - rr.auprc),
            })
    pd.DataFrame(rows).to_csv(OUT / "matched_size_paired_differences.tsv", sep="\t", index=False)

    expected_rows = len(studies_order) * len(panels) * len(models)
    # One fold audit row records the exact-size assertion for all four panels;
    # the long membership table carries the per-taxon panel rows.
    expected_membership_rows = len(studies_order)
    # For long membership, each panel has nICP rows.  This explicit count
    # catches a truncated or duplicated membership export.
    expected_long_membership_rows = int(audits.nICP.sum() * len(panels)) if not audits.empty else 0
    observed_long_membership_rows = len(memberships)
    receipt = {
        "status": "PASS" if (not errors and len(metrics) == expected_rows and len(audits) == len(studies_order) and len(memberships) == expected_long_membership_rows and (audits["ARP_nICP_n"] == audits["nICP"]).all() and (audits["DCP_nICP_n"] == audits["nICP"]).all() and (audits["ICP_n"] == audits["nICP"]).all() and (audits["random_nICP_n"] == audits["nICP"]).all()) else "PARTIAL",
        "smoke": bool(args.smoke),
        "folds": len(studies_order),
        "models": models,
        "panels": panels,
        "expected_metric_rows": expected_rows,
        "observed_metric_rows": len(metrics),
        "expected_fold_audit_rows": expected_membership_rows,
        "observed_fold_audit_rows": len(audits),
        "expected_panel_membership_groups": len(studies_order) * len(panels),
        "observed_panel_membership_groups": len(membership_groups),
        "expected_long_membership_rows": expected_long_membership_rows,
        "observed_long_membership_rows": observed_long_membership_rows,
        "bootstrap_replicates": BOOT,
        "bootstrap_valid_min": int(metrics.bootstrap_valid.min()) if not metrics.empty else 0,
        "bootstrap_valid_max": int(metrics.bootstrap_valid.max()) if not metrics.empty else 0,
        "matrix_sha256": sha256(MATRIX),
        "metadata_sha256": sha256(META),
        "association_sha256": {str(p.name): sha256(p) for p in sorted(NESTED.glob("fold_*_association.tsv"))},
        "membership_input_sha256": sha256(NESTED / "nested_loso_panel_membership_long.tsv"),
        "reference_metrics_sha256": sha256(NESTED / "nested_loso_metrics.tsv"),
        "selection": "ARP_nICP uses saved fold q ranking; DCP_nICP uses saved q ranking after q<0.10 and direction_agreement>=0.60; ICP and random panels are imported from saved nested membership; all exact-size panels equal observed ICP n per outer fold.",
        "claim_boundary": "Exact-size matched refits are a nested internal sensitivity analysis. They isolate panel-size effects from association/direction/ICP criteria and are not independent external validation or clinical validation.",
        "errors": errors,
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "matched_size_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    (OUT / "matched_size_claim_boundary.md").write_text(
        "# Exact-size matched refit claim boundary\n\n"
        "ARP_nICP and DCP_nICP are fold-specific panels truncated to the observed train-only ICP size. "
        "ICP and prevalence-matched random membership are imported from the saved nested LOSO run. "
        "The outer held-out study is excluded from preprocessing, ranking, size selection, and model fitting. "
        "These results isolate panel-size effects within the internal nested transferability audit; they do not establish independent external or clinical validation.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

