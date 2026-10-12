"""Read-only source adapters for the deposited MP4 coordinate system."""
from pathlib import Path

import numpy as np
import pandas as pd

from .boundary import read_tsv, sha

COHORTS = {
    "LiuNN_2022": ("Public_study__LiuNN_2022", {"CRC":78,"Control":85}, "primary"),
    "YangY_2021": ("Public_study__YangY_2021", {"CRC":100,"Control":100}, "primary"),
    "IIGM_TU": ("This_study_cohort6__IIGM_TU", {"CRC":18,"Control":39}, "primary"),
    "YangJ_2020": ("Public_study__YangJ_2020", {"CRC":98,"Control":95}, "secondary"),
    "NHSII": ("This_study_cohort5__NSHII", {"CRC":14,"Control":448,"Adenoma":435}, "secondary"),
    "IIGM_CZ": ("This_study_cohort3__ONCOBIOME_IIGM_CZ", {"CRC":59,"Control":46,"Adenoma":19}, "secondary"),
}
PRIMARY = ("LiuNN_2022", "YangY_2021", "IIGM_TU")
EXPECTED = (1257, 632, 625, 9)


def source_path(config, stem, kind):
    if not stem or Path(stem).name != stem:
        raise ValueError("Invalid source stem")
    return config["source_root"] / "extracted" / kind / (stem + ".tsv")


def verify_source(config, path):
    """Check extracted bytes against the existing public recovery contract."""
    contract = Path(__file__).resolve().parents[1] / "r2_reproducibility/input_contract.tsv"
    frame = read_tsv(contract, keep_default_na=False)
    relative = Path(path).resolve().relative_to(config["source_root"].resolve())
    key = "source_root/" + relative.as_posix()
    row = frame.loc[frame.input_key.eq(key)]
    if len(row) != 1 or sha(path) != row.sha256.iloc[0]:
        raise ValueError(f"Source is not the pinned deposited MP4 input: {relative}")
    return Path(path)


def profile(config, stem, sample_ids=None):
    path = source_path(config, stem, "metaphlan4_profiles")
    verify_source(config, path)
    frame = read_tsv(path, index_col=0, keep_default_na=False)
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("Duplicate source feature/sample axis")
    frame = frame.loc[frame.index.str.rsplit("|", n=1).str[-1].str.startswith("s__")]
    ids = frame.columns.tolist() if sample_ids is None else list(sample_ids)
    if len(ids) != len(set(ids)) or not set(ids).issubset(frame.columns):
        raise ValueError("Missing or duplicate entire sample profiles")
    frame = frame.loc[:, ids].T.astype(float)
    values = frame.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid deposited profile abundance")
    if not frame.sum(axis=1).between(99.9,100.1).all():
        raise ValueError("Deposited profiles must be percentage abundance")
    return frame


def load_training(config):
    meta = read_tsv(config["training_manifest"], keep_default_na=False)
    fields = ["sample_uid","sample_id","subject_id","study_name","source_stem","study_condition"]
    if not set(fields).issubset(meta.columns) or meta[fields].isna().any().any() or meta[fields].eq("").any().any():
        raise ValueError("Missing training identities or labels")
    if meta.sample_uid.duplicated().any() or meta.duplicated(["study_name","subject_id"]).any():
        raise ValueError("Duplicate training identity or within-study subject")
    actual = (len(meta), int(meta.study_condition.eq("CRC").sum()),
              int(meta.study_condition.eq("control").sum()), meta.study_name.nunique())
    if actual != EXPECTED or set(meta.study_condition) != {"CRC","control"}:
        raise ValueError(f"Nine-study clinical counts differ: {actual} versus {EXPECTED}")
    if not meta.groupby("study_name").study_condition.nunique().eq(2).all():
        raise ValueError("Each study requires both outcome groups")
    blocks, qc, inputs = [], [], [config["training_manifest"]]
    for stem, group in meta.groupby("source_stem", sort=True):
        block = profile(config, stem, group.sample_id)
        block.index = group.set_index("sample_id").loc[block.index,"sample_uid"]
        path = source_path(config, stem, "metadata")
        verify_source(config, path)
        source_meta = read_tsv(path, keep_default_na=False)
        if source_meta["Sample Source ID"].duplicated().any():
            raise ValueError("Duplicate source metadata sample")
        labels = source_meta.set_index("Sample Source ID").loc[group.sample_id,"Study condition"].str.lower().to_numpy()
        if not np.array_equal(labels, group.study_condition.str.lower().to_numpy()):
            raise ValueError("Training labels disagree with deposited source")
        blocks.append(block)
        qc.append({"source_stem":stem,"n":len(block),"species_rows":block.shape[1],
                   "sum_min":block.sum(axis=1).min(),"sum_max":block.sum(axis=1).max(),
                   "metadata_labels_match":True})
        inputs.extend([path,source_path(config,stem,"metaphlan4_profiles")])
    universe = sorted(set().union(*(set(block.columns) for block in blocks)))
    raw = pd.concat([b.reindex(columns=universe,fill_value=0.) for b in blocks]).loc[meta.sample_uid]
    return meta, raw, pd.DataFrame(qc), inputs
