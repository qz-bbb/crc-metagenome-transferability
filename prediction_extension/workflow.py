"""Bounded runners; every invocation has one fresh destination and action."""
import argparse
import itertools
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .boundary import (check_receipt, fresh, load_config, point_replay, protected_paths,
                       read_tsv, save_json, save_table, seal, sha)
from .core import DEFINITIONS, concordance, effect_definitions
from .evaluation import (CLASSIFIERS, DEFAULT_MODELS, PANELS, evaluate_study,
                         summarize_studies, validate_predictions)
from .external import attach_outcomes, compare_baseline, transform_blind
from .external_evaluation import evaluate_cohort, summarize_external
from .models import (ModelConvergenceError, fit_panel, make_model, model_fingerprint,
                     selected_features)
from .selection import (DEFAULT_CONFIG, build_training_state, package_versions,
                        fingerprint,
                        split_outer_fold, training_input_fingerprint,
                        validate_training, verify_training_state_integrity)
from .source import COHORTS, PRIMARY, load_training, profile, source_path, verify_source


def save_evaluation(directory, result):
    save_table(directory / "metrics.tsv", result["metrics"])
    np.savez_compressed(directory / "scores.npz", point=result["point"],
                        bootstrap=result["bootstrap"])
    save_json(directory / "evaluation.json", {"model_keys":result["model_keys"],
              "index_hash":result["index_hash"],"seed":20261011,"n_boot":len(result["bootstrap"])})


def load_evaluation(directory):
    directory = Path(directory)
    data = json.loads((directory / "evaluation.json").read_text())
    with np.load(directory / "scores.npz", allow_pickle=False) as arrays:
        point, bootstrap = arrays["point"].copy(), arrays["bootstrap"].copy()
    return {**data, "point":point, "bootstrap":bootstrap,
            "metrics":read_tsv(directory / "metrics.tsv")}


def write_state(directory, state):
    save_table(directory / "feature_evidence.tsv", state["evidence"].rename_axis("taxon").reset_index())
    save_table(directory / "direction_details.tsv", state["direction_details"])
    rows = []
    for panel, members in state["memberships"].items():
        family, _, definition = panel.partition("_")
        for rank, taxon in enumerate(members,1):
            rows.append({"panel":panel,"family":family,"definition":definition or "none",
                         "rank":rank,"taxon":taxon,"k":state["k"],
                         "natural_count":state["eligible_counts"].get(panel,len(state["features"]))})
    save_table(directory / "panel_membership.tsv",pd.DataFrame(rows,columns=[
        "panel","family","definition","rank","taxon","k","natural_count"]))
    save_json(directory / "state_summary.json",{k:state[k] for k in
              ["features","pseudocount","eligible_counts","k","hashes"]})
    joblib.dump(state,directory / "state.joblib",compress=3)


def read_state(path):
    check_receipt(Path(path).parent)
    state = joblib.load(path)
    verify_training_state_integrity(state)
    if set(state["memberships"]) != set(PANELS):
        raise ValueError("Nine-panel membership grid required")
    return state


def membership_summary(states):
    counts, pairs, changes, reasons = [], [], [], []
    training_overlap, partitions = [], []
    for scope, state in sorted(states.items()):
        evidence = state["evidence"]
        if evidence.index.name != "taxon":
            evidence = evidence.rename_axis("taxon")
        selected = {p:set(state["memberships"][p]) for p in PANELS}
        eligible = {p:set(evidence.index[evidence["eligible_"+p]]) for p in PANELS if p != "ARP"}
        for pool, sets in [("eligible",eligible),("selected",selected)]:
            for left,right in itertools.combinations(sets,2):
                intersection,union = len(sets[left]&sets[right]),len(sets[left]|sets[right])
                pairs.append({"held_out_study":scope,"pool":pool,"left":left,"right":right,
                              "n_left":len(sets[left]),"n_right":len(sets[right]),
                              "intersection":intersection,"union":union,
                              "jaccard":intersection/union if union else np.nan,
                              "identical":sets[left]==sets[right]})
        for panel in PANELS:
            counts.append({"held_out_study":scope,"panel":panel,"retained_features":len(evidence),
                           "natural_eligible":state["eligible_counts"].get(panel,len(evidence)),
                           "selected_n":len(selected[panel]),"k":state["k"]})
        for family in ("DCP","ICP"):
            baseline = selected[family+"_raw_median"]
            for definition in DEFINITIONS:
                current = selected[family+"_"+definition]
                union = len(current|baseline)
                training_overlap.append({"held_out_study":scope,"family":family,
                    "definition":definition,"k":state["k"],
                    "intersection_with_raw_median":len(current&baseline),
                    "jaccard_with_raw_median":len(current&baseline)/union if union else np.nan,
                    "natural_candidate_count":state["eligible_counts"][family+"_"+definition]})
        if "partition_summary" in state:
            partitions.append(state["partition_summary"])
        for family in ("DCP","ICP"):
            baseline = eligible[family+"_raw_median"]
            for definition in ("raw_mean","clr_mean","clr_median"):
                current = eligible[family+"_"+definition]
                for taxon in sorted(baseline ^ current):
                    row = evidence.loc[taxon]
                    changes.append({"held_out_study":scope,"family":family,"alternative":definition,
                        "taxon":taxon,"change":"added" if taxon in current else "removed",
                        "q":row.q,"prevalence":row.prevalence,
                        "raw_median_agreement":row.direction_raw_median,
                        "alternative_agreement":row["direction_"+definition],
                        "selected_baseline":taxon in selected[family+"_raw_median"],
                        "selected_alternative":taxon in selected[family+"_"+definition]})
        excluded = evidence.loc[(evidence.q < .1) & (evidence.direction_raw_median < .6),["prevalence"]]
        details = state["direction_details"]
        rows = details.loc[details.definition.eq("raw_median")].merge(
            excluded.reset_index(),on="taxon",validate="many_to_one")
        denominator = len(set(state["training_studies"]))
        if len(rows) != len(excluded)*denominator:
            raise ValueError("Direction study denominator changed")
        from .membership import direction_category
        rows["category"] = [direction_category(*values) for values in rows[
            ["detected","tolerance_zero","same_direction","reverse_direction"]].itertuples(index=False,name=None)]
        rows["prevalence_bin"] = pd.cut(rows.prevalence,[0,.1,.25,.5,1],
            labels=["<=0.10","0.10-0.25","0.25-0.50",">0.50"],include_lowest=True)
        for group, part in [("all",rows)]+[(str(label),p) for label,p in rows.groupby("prevalence_bin",observed=True)]:
            for category in ["not_detected","zero_difference","opposite","concordant"]:
                n = int(part.category.eq(category).sum())
                reasons.append({"held_out_study":scope,"prevalence_bin":group,
                    "excluded_taxa":part.taxon.nunique(),"taxon_study_denominator":len(part),
                    "category":category,"count":n,"fraction":n/len(part) if len(part) else np.nan})
    result = {"panel_counts.tsv":pd.DataFrame(counts),"panel_overlap.tsv":pd.DataFrame(pairs),
            "changed_eligible_taxa.tsv":pd.DataFrame(changes),
            "raw_median_exclusion_breakdown.tsv":pd.DataFrame(reasons),
            "training_membership_overlap.tsv":pd.DataFrame(training_overlap)}
    if partitions:
        result["fold_summary.tsv"] = pd.DataFrame(partitions)
    return result


def direction_context(raw, labels, state, cohort, role, historical=None):
    features = state["features"]
    evidence = state["evidence"].loc[features]
    values = raw.reindex(columns=features,fill_value=0.)
    effects, detected = effect_definitions(values,labels,state["pseudocount"])
    groups = {"ALL800":np.ones(len(features),dtype=bool)}
    groups.update({"selected__"+p:np.asarray([t in members for t in features])
                   for p,members in state["memberships"].items()})
    groups.update({"natural__"+p:evidence["eligible_"+p].to_numpy(bool)
                   for p in state["eligible_counts"]})
    records, summary = [], []
    old = None
    if historical is not None:
        old = historical.set_index("taxon")
        if old.index.has_duplicates or set(old.index) != set(features):
            raise ValueError("Historical external direction universe differs")
        old = old.loc[features]
        for column,current in [("training_coef",evidence.coef.to_numpy()),
                               ("training_q",evidence.q.to_numpy()),
                               ("external_mean_clr_difference",effects["clr_mean"]),
                               ("external_median_relative_difference",effects["raw_median"])]:
            if not np.allclose(current,old[column].to_numpy(),atol=1e-10,rtol=1e-8):
                raise ValueError(f"Historical directional replay changed: {column}")
    for definition in DEFINITIONS:
        effect = effects[definition]
        same = concordance(effect,evidence.coef.to_numpy(),detected,1e-12)
        zero = np.abs(effect)<=1e-12
        opposite = detected & ~zero & (np.sign(effect)!=np.sign(evidence.coef.to_numpy()))
        if old is not None and definition == "clr_mean" and not np.array_equal(same,old.concordant_clr.to_numpy(bool)):
            raise ValueError("Historical detection-aware concordance changed")
        frame = pd.DataFrame({"cohort":cohort,"taxon":features,"external_definition":definition,
            "training_coef":evidence.coef.to_numpy(),"training_q":evidence.q.to_numpy(),
            "external_effect":effect,"detected":detected,"zero_difference":zero,
            "concordant_detected":same,"opposite_detected":opposite,
            "explicit_profile_row":[t in raw.columns for t in features]})
        if old is not None:
            frame["external_q_full800_CLR_mean"] = old.external_q_full_800.to_numpy()
        for name,mask in groups.items():
            frame[name] = mask
            n = int(mask.sum())
            summary.append({"cohort":cohort,"role":role,"group":name,
                "external_definition":definition,"denominator":n,"detected":int((mask&detected).sum()),
                "not_detected":int((mask&~detected).sum()),"zero_difference":int((mask&zero).sum()),
                "concordant_detected":int((mask&same).sum()),
                "opposite_detected":int((mask&opposite).sum()),
                "concordance_fraction":float((mask&same).sum()/n) if n else np.nan})
        records.append(frame)
    return pd.DataFrame(summary),pd.concat(records,ignore_index=True)


def fitted_one(state, raw, labels, panel, classifier):
    verify_training_state_integrity(state)
    _, y, studies, ids = validate_training(raw,labels,state["training_studies"],state["sample_ids"])
    versions = package_versions()
    if versions != state["package_versions"] or training_input_fingerprint(
            raw,y,studies,ids,state["config"],versions) != state["hashes"]["training_inputs"]:
        raise ValueError("Training/runtime drift from frozen selection")
    family,_,definition = panel.partition("_")
    members = selected_features(state,family,definition or None)
    indices = [state["features"].index(t) for t in members]
    from .core import clr
    full = clr(raw.loc[:,state["features"]],state["pseudocount"])
    model = make_model(classifier)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            model.fit(full[:,indices],y)
    except ConvergenceWarning as exc:
        raise ModelConvergenceError(str(exc)) from exc
    return {"pipeline":model,"indices":indices,"features":members,
            "fingerprint":model_fingerprint(state,classifier,family,definition or None)}


def _blind_complete(config):
    directories = config.get("blind_dirs",[])
    if len(directories) != 6 or len(set(directories)) != 6:
        raise ValueError("Six sealed blind cohort directories required before attaching labels")
    receipts = [check_receipt(p) for p in directories]
    if {r.get("cohort") for r in receipts} != set(COHORTS) or any(r["action"]!="external-predict" for r in receipts):
        raise ValueError("Missing/extra blind cohort receipts")
    locks = {r.get("model_bundle_sha256") for r in receipts}
    if len(locks) != 1 or None in locks:
        raise ValueError("Blind cohorts must share one model lock")
    if "model_bundle" in config and locks != {sha(config["model_bundle"])}:
        raise ValueError("Configured model lock differs from blind predictions")
    known_ids,known_hashes = set(),set()
    if "model_bundle" in config:
        bundle = _bundle(config["model_bundle"])
        known_ids.update(bundle["training_sample_ids"])
        known_hashes.update(bundle["training_vector_hashes"])
    for path in directories:
        table = read_tsv(path/"sample_vector_hashes.tsv",keep_default_na=False)
        if (not {"sample_id","vector_sha256"}.issubset(table.columns) or table.empty
                or table.sample_id.duplicated().any() or table.vector_sha256.duplicated().any()
                or set(table.sample_id)&known_ids or set(table.vector_sha256)&known_hashes):
            raise ValueError("Complete blind cohort sample/vector overlap")
        known_ids.update(table.sample_id)
        known_hashes.update(table.vector_sha256)
    return {r["cohort"]:p for r,p in zip(receipts,directories)}


def _external_identity(receipt):
    identity = {key:receipt.get(key) for key in ("model_bundle_sha256","blind_set_sha256")}
    if any(not isinstance(value,str) or len(value)!=64 or
           any(character not in "0123456789abcdef" for character in value)
           for value in identity.values()):
        raise ValueError("Missing or invalid external evaluation identity")
    return identity


def _blind_identity(directories):
    if set(directories)!=set(COHORTS):
        raise ValueError("Complete six-cohort blind identity required")
    receipts = {cohort:check_receipt(path) for cohort,path in directories.items()}
    locks = {receipt.get("model_bundle_sha256") for receipt in receipts.values()}
    if len(locks)!=1 or any(receipt.get("action")!="external-predict" or
                           receipt.get("cohort")!=cohort for cohort,receipt in receipts.items()):
        raise ValueError("Mixed or mislabeled blind model identity")
    lock = next(iter(locks))
    blind_set = fingerprint({"model_bundle_sha256":lock,"cohort_receipts":{
        cohort:sha(directories[cohort]/"receipt.json") for cohort in sorted(COHORTS)}})
    return _external_identity({"model_bundle_sha256":lock,"blind_set_sha256":blind_set})


def _bundle(path):
    check_receipt(Path(path).parent)
    bundle = joblib.load(path)
    verify_training_state_integrity(bundle["state"])
    if set(bundle["models"]) != {f"{p}::{c}" for p,c in DEFAULT_MODELS}:
        raise ValueError("Model lock requires all 18 model configurations")
    return bundle


def _inputs(config):
    values = [v for key,v in config.items() if isinstance(v,Path) and key not in {"output_root","source_root"}]
    return [p for p in values if p.is_file()]


def _initial_hashes(config):
    paths = set(_inputs(config))
    if "source_root" in config:
        paths.update((config["source_root"]/"extracted").rglob("*.tsv"))
    for key in ("evaluation_dirs","state_dirs","blind_dirs","model_dirs"):
        for directory in config.get(key,[]):
            paths.update(p for p in directory.iterdir() if p.is_file())
    for key in ("state","model_bundle"):
        if key in config:
            receipt = config[key].parent/"receipt.json"
            if receipt.exists():
                paths.add(receipt)
    return {str(p.resolve()):sha(p) for p in paths}


def run(args, config):
    # Validate and reserve a fresh action directory before any file is written.
    initial_hashes = _initial_hashes(config)
    directory = fresh(config["output_root"],protected_paths(config))
    inputs = _inputs(config)
    details = {}
    action = args.action
    if action in {"audit","select","fit-model"}:
        meta,raw,qc,source_inputs = load_training(config)
        inputs.extend(source_inputs)
        labels,studies,ids = meta.study_condition.eq("CRC").astype(int).to_numpy(),meta.study_name.to_numpy(),meta.sample_uid.tolist()
        if action == "audit":
            save_table(directory/"metadata.tsv",meta[["sample_uid","study_name","study_condition"]])
            save_table(directory/"source_profile_QC.tsv",qc)
            save_table(directory/"study_inventory.tsv",meta.groupby(["study_name","study_condition"]).size().rename("n").reset_index())
            details.update(training_n=len(meta),crc=int(labels.sum()),control=int((labels==0).sum()),
                           studies=sorted(meta.study_name.unique()),external_labels_read=False)
        else:
            if action == "select":
                heldout = None
                train = {"raw":raw,"y":labels,"studies":studies,"sample_ids":ids}
                if args.fold is not None:
                    names = sorted(meta.study_name.unique())
                    if not 0 <= args.fold < 9:
                        raise ValueError("Fold index must lie in 0..8")
                    heldout = names[args.fold]
                    train = split_outer_fold(raw,labels,studies,ids,heldout)["train"]
                state = build_training_state(train["raw"],train["y"],train["studies"],train["sample_ids"],DEFAULT_CONFIG)
                state["held_out_study"] = heldout
                state["all_study_names"] = sorted(meta.study_name.unique())
                if heldout is not None:
                    test_mask = studies==heldout
                    state["partition_summary"] = {"held_out_study":heldout,
                        "training_n":len(train["y"]),"test_n":int(test_mask.sum()),
                        "test_crc":int(labels[test_mask].sum()),"feature_count":len(state["features"]),
                        "pseudocount":state["pseudocount"],"k":state["k"],
                        "prediction_rows":int(test_mask.sum())*18}
                write_state(directory,state)
                details.update(k=state["k"],training_state=state["hashes"]["training_state"],held_out_study=heldout,
                               selection_scope="full_training" if heldout is None else "outer_fold",
                               external_arrays_or_labels_used=False)
            else:
                if args.panel is None or args.classifier is None:
                    raise ValueError("fit-model requires --panel and --classifier")
                state = read_state(config["state"])
                heldout = state.get("held_out_study")
                if heldout is None:
                    fitted = fitted_one(state,raw,labels,args.panel,args.classifier)
                    joblib.dump(fitted,directory/"model.joblib",compress=3)
                else:
                    split = split_outer_fold(raw,labels,studies,ids,heldout)
                    train,test = split["train"],split["test"]
                    family,_,definition = args.panel.partition("_")
                    p = fit_panel(state,train["raw"],train["y"],test["raw"],args.classifier,family,definition or None)
                    save_table(directory/"predictions.tsv",pd.DataFrame({
                        "sample_uid":test["sample_ids"],"held_out_study":heldout,"panel":args.panel,
                        "classifier":args.classifier,"k":state["k"],"y":test["y"],"probability_crc":p}))
                details.update(panel=args.panel,classifier=args.classifier,held_out_study=heldout,
                               training_state=state["hashes"]["training_state"])
    elif action == "assemble":
        state = read_state(config["state"])
        model_dirs = config.get("model_dirs",[])
        if len(model_dirs) != 18 or len(set(model_dirs)) != 18:
            raise ValueError("Exactly 18 distinct fit-model directories required")
        receipts = [check_receipt(p) for p in model_dirs]
        keys = [(r.get("panel"),r.get("classifier")) for r in receipts]
        if set(keys) != set(DEFAULT_MODELS) or any(r["action"]!="fit-model" or
            r.get("training_state")!=state["hashes"]["training_state"] or
            r.get("held_out_study")!=state.get("held_out_study") for r in receipts):
            raise ValueError("Fit grid has missing models or divergent selection state")
        inputs.extend(p/"receipt.json" for p in model_dirs)
        if state.get("held_out_study") is None:
            meta,raw,_,source_inputs = load_training(config)
            inputs.extend(source_inputs)
            vectors = raw.loc[:,state["features"]].to_numpy(dtype="<f8")
            import hashlib
            hashes = [hashlib.sha256(v.tobytes()).hexdigest() for v in vectors]
            bundle = {"state":state,"models":{f"{p}::{c}":joblib.load(d/"model.joblib")
                      for (p,c),d in zip(keys,model_dirs)},"training_sample_ids":meta.sample_id.tolist(),
                      "training_vector_hashes":hashes}
            joblib.dump(bundle,directory/"bundle.joblib",compress=3)
            details.update(models=18,training_state=state["hashes"]["training_state"])
        else:
            pred = pd.concat([read_tsv(p/"predictions.tsv") for p in model_dirs],ignore_index=True)
            meta,_,_,source_inputs = load_training(config)
            inputs.extend(source_inputs)
            meta = meta.loc[meta.study_name.eq(state["held_out_study"]),["sample_uid","study_name","study_condition"]]
            validate_predictions(pred,meta)
            save_table(directory/"predictions.tsv",pred)
            save_table(directory/"metadata.tsv",meta)
            details.update(held_out_study=state["held_out_study"],models=18,prediction_rows=len(pred))
    elif action == "external-predict":
        cohort = args.cohort
        if cohort is None:
            raise ValueError("--cohort required")
        bundle = _bundle(config["model_bundle"])
        state = bundle["state"]
        stem,counts,_ = COHORTS[cohort]
        raw = profile(config,stem)
        if len(raw) != sum(counts.values()) or set(raw.index)&set(bundle["training_sample_ids"]):
            raise ValueError("External source sample count/identity drift")
        import hashlib
        vectors = raw.reindex(columns=state["features"],fill_value=0.).to_numpy(dtype="<f8")
        hashes = [hashlib.sha256(v.tobytes()).hexdigest() for v in vectors]
        prior_ids,prior_hashes = set(),set(bundle["training_vector_hashes"])
        for prior in config.get("blind_dirs",[]):
            r = check_receipt(prior)
            if r.get("action")!="external-predict" or r.get("model_bundle_sha256")!=sha(config["model_bundle"]):
                raise ValueError("Prior blind cohort used different model lock")
            table = read_tsv(prior/"sample_vector_hashes.tsv",keep_default_na=False)
            prior_ids.update(table.sample_id)
            prior_hashes.update(table.vector_sha256)
            inputs.append(prior/"receipt.json")
        if set(raw.index)&prior_ids or len(set(hashes))!=len(hashes) or set(hashes)&prior_hashes:
            raise ValueError("Training/cross-external sample ID or exact retained-vector overlap")
        transformed,missing = transform_blind(raw,state,same_source_verified=True)
        rows = []
        for key,item in bundle["models"].items():
            panel,classifier = key.split("::")
            p = item["pipeline"].predict_proba(transformed[:,item["indices"]])[:,1]
            if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
                raise ValueError("Invalid blind probabilities")
            rows.extend({"sample_id":sid,"panel":panel,"classifier":classifier,
                         "k":state["k"],"probability_crc":float(prob)} for sid,prob in zip(raw.index,p))
        save_table(directory/"predictions_blind.tsv",pd.DataFrame(rows))
        if "baseline_predictions" in config:
            replay = compare_baseline(pd.DataFrame(rows),read_tsv(config["baseline_predictions"]))
            save_table(directory/"baseline_probability_replay.tsv",replay)
        save_table(directory/"sample_vector_hashes.tsv",pd.DataFrame({"sample_id":raw.index,"vector_sha256":hashes}))
        save_table(directory/"unreported_taxon_rows.tsv",pd.DataFrame({"taxon":missing}))
        coverage = [{"panel":"ALL800","features":len(state["features"]),"explicit_rows":len(state["features"])-len(missing),"unreported_rows":len(missing)}]
        for panel,members in state["memberships"].items():
            absent = sum(t in missing for t in members)
            coverage.append({"panel":panel,"features":len(members),"explicit_rows":len(members)-absent,"unreported_rows":absent})
        save_table(directory/"feature_coverage.tsv",pd.DataFrame(coverage))
        inputs.append(source_path(config,stem,"metaphlan4_profiles"))
        details.update(cohort=cohort,models=18,metadata_labels_read_by_prediction=False,
                       model_bundle_sha256=sha(config["model_bundle"]),inference_status="post_outcome_sensitivity",
                       identity_limitation="43 training identities remain accession-unresolved aliases")
    elif action in {"external-evaluate","directions"}:
        cohort = args.cohort
        if cohort is None:
            raise ValueError("--cohort required")
        blind_dirs = _blind_complete(config)
        details.update(_blind_identity(blind_dirs))
        inputs.extend(p/"receipt.json" for p in blind_dirs.values())
        stem,counts,role = COHORTS[cohort]
        path = source_path(config,stem,"metadata")
        verify_source(config,path)
        clinical = read_tsv(path,keep_default_na=False)
        inputs.append(path)
        pred,meta = attach_outcomes(read_tsv(blind_dirs[cohort]/"predictions_blind.tsv"),clinical,cohort,counts)
        if action == "external-evaluate":
            result = evaluate_cohort(pred,meta,sorted(COHORTS).index(cohort),n_boot=2000,seed=20261011)
            save_table(directory/"predictions.tsv",pred)
            save_table(directory/"metadata.tsv",meta)
            save_evaluation(directory,result)
        else:
            state = _bundle(config["model_bundle"])["state"]
            clinical = clinical.loc[clinical["Study condition"].isin(["CRC","Control"])]
            raw = profile(config,stem,clinical["Sample Source ID"])
            history = read_tsv(config["historical_directions"])
            summary,context = direction_context(raw,clinical["Study condition"].eq("CRC").to_numpy(int),
                                               state,cohort,role,history)
            save_table(directory/"direction_summary.tsv",summary)
            save_table(directory/(cohort+"_all800_four_definition_context.tsv"),context)
            inputs.append(source_path(config,stem,"metaphlan4_profiles"))
        details.update(cohort=cohort,inference_status="post_outcome_sensitivity",models_modified=False,
                       seed=20261011,bootstrap_context=2,n_boot=2000)
    elif action in {"point-replay","evaluate"}:
        pred,meta = read_tsv(config["predictions"]),read_tsv(config["metadata"],keep_default_na=False)
        if action == "point-replay":
            save_table(directory/"point_metrics.tsv",point_replay(pred,meta))
        else:
            names = config.get("study_names")
            if not isinstance(names,list) or len(names)!=9 or len(set(names))!=9 or names!=sorted(names):
                raise ValueError("Sorted nine-study names must be supplied explicitly")
            if args.fold is None or not 0 <= args.fold < 9:
                raise ValueError("--fold in 0..8 required")
            study = names[args.fold]
            pred = pred.loc[pred.held_out_study.eq(study)]
            meta = meta.loc[meta.study_name.eq(study)]
            save_table(directory/"predictions.tsv",pred)
            save_table(directory/"metadata.tsv",meta)
            save_evaluation(directory,evaluate_study(pred,meta,args.fold,n_boot=2000,seed=20261011))
            details.update(study=study,seed=20261011,bootstrap_context=1,n_boot=2000)
    elif action in {"summary-internal","summary-external"}:
        expected = config.get("study_names") if action=="summary-internal" else list(COHORTS)
        if action=="summary-internal" and (not isinstance(expected,list) or expected!=sorted(expected)):
            raise ValueError("Internal summary requires the sorted study-name list")
        directories = config.get("evaluation_dirs",[])
        if expected is None or len(directories)!=len(expected) or len(set(directories))!=len(expected):
            raise ValueError("Complete distinct evaluation directories required")
        results = {}
        shared_external_identity = None
        for path in directories:
            receipt = check_receipt(path)
            result = load_evaluation(path)
            context = 1 if action=="summary-internal" else 2
            evaluator = "evaluate" if context==1 else "external-evaluate"
            if (receipt.get("action")!=evaluator or receipt.get("seed")!=20261011
                    or receipt.get("bootstrap_context")!=context or receipt.get("n_boot")!=2000
                    or result.get("seed")!=20261011 or result.get("n_boot")!=2000
                    or result["bootstrap"].shape[0]!=2000):
                raise ValueError("Protocol summaries require the fixed seed/context and 2000 draws")
            names = result["metrics"].study_name.unique()
            if len(names)!=1 or names[0] in results:
                raise ValueError("Duplicate or mixed evaluation scope")
            if action=="summary-external":
                if receipt.get("cohort")!=names[0]:
                    raise ValueError("External receipt cohort disagrees with metric scope")
                identity = _external_identity(receipt)
                if shared_external_identity is not None and identity!=shared_external_identity:
                    raise ValueError("Mixed external model lock or complete blind-set identity")
                shared_external_identity = identity
            results[names[0]] = result
            inputs.append(path/"receipt.json")
        if action=="summary-internal":
            metrics,contrasts = summarize_studies(results,expected)
        else:
            details.update(shared_external_identity)
            metrics,contrasts = summarize_external(results,expected,PRIMARY)
        save_table(directory/"macro_metrics.tsv",metrics)
        save_table(directory/"paired_contrasts.tsv",contrasts)
        combined_metrics = pd.concat([results[name]["metrics"] for name in expected],ignore_index=True)
        save_table(directory/"study_metrics.tsv",combined_metrics)
        if action=="summary-internal":
            save_table(directory/"primary_comparisons.tsv",contrasts.loc[
                contrasts.scope.eq("MACRO") & contrasts.primary])
        else:
            save_table(directory/"primary3_macro_metrics.tsv",metrics)
            save_table(directory/"all6_cohort_metrics.tsv",combined_metrics)
            save_table(directory/"paired_contrasts_all_scopes.tsv",contrasts)
        predictions = pd.concat([read_tsv(p/"predictions.tsv") for p in directories],ignore_index=True)
        metadata = pd.concat([read_tsv(p/"metadata.tsv",keep_default_na=False) for p in directories],ignore_index=True)
        validate_predictions(predictions,metadata)
        save_table(directory/"all_predictions.tsv",predictions)
        save_table(directory/"all_metadata.tsv",metadata)
    elif action == "membership":
        states = {}
        for path in config.get("state_dirs",[]):
            state = read_state(path/"state.joblib")
            name = state.get("held_out_study") or "full_training"
            if name in states:
                raise ValueError("Duplicate membership scope")
            states[name] = state
            inputs.append(path/"receipt.json")
        if len(states) not in {1,9}:
            raise ValueError("Supply full-training state or all nine fold states")
        if len(states)==1 and next(iter(states))!="full_training":
            raise ValueError("Single state must be full-training")
        for name,frame in membership_summary(states).items():
            save_table(directory/name,frame)
    else:
        raise ValueError("Unsupported action")
    # Recheck initial identities; no concurrently edited input may earn PASS.
    details["packages"] = package_versions()
    return seal(directory,action,sorted(set(inputs)),details,expected_inputs=initial_hashes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("action",choices=["audit","select","fit-model","assemble","external-predict",
        "external-evaluate","evaluate","summary-internal","summary-external","membership",
        "directions","point-replay"])
    parser.add_argument("--fold",type=int)
    parser.add_argument("--cohort",choices=list(COHORTS))
    parser.add_argument("--panel",choices=list(PANELS))
    parser.add_argument("--classifier",choices=list(CLASSIFIERS))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    # No action automatically starts another stage or loops over folds/cohorts.
    with threadpool_limits(limits=1):
        result = run(args,config)
    print(json.dumps({"status":result["status"],"action":args.action,"outputs":len(result["outputs"])}))
    return result
