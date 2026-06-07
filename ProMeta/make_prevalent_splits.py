import argparse
import json
import os
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_eid(value):
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def to_timestamp(value):
    ts = pd.Timestamp(value)
    return None if pd.isna(ts) else ts


def load_pickle(path):
    with open(path, "rb") as handle:
        return pkl.load(handle)


def dump_pickle(path, value):
    with open(path, "wb") as handle:
        pkl.dump(value, handle, protocol=pkl.HIGHEST_PROTOCOL)


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def add_to_split(target_cases, target_controls, term, record, case_key, control_key):
    target_cases[term] = record[case_key]
    target_controls[term] = record[control_key]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate term2pre train/valid/test split files for ProMeta."
    )
    parser.add_argument("--data_dir", required=True, help="Directory containing term2caseids.pkl, term2casedates.pkl, and term2controlids.pkl.")
    parser.add_argument("--proteomics_date_csv", required=True, help="CSV with EID and proteomics_test_date columns.")
    parser.add_argument("--output_dir", default=None, help="Directory to write split files. Defaults to data_dir.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_cases", type=int, default=50)
    parser.add_argument("--max_cases", type=int, default=1000)
    parser.add_argument("--train_min_cases", type=int, default=32)
    parser.add_argument("--heldout_fraction", type=float, default=0.1)
    parser.add_argument("--jaccard_threshold", type=float, default=0.75)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = [
        "term2pre_cases_train.pkl",
        "term2pre_controls_train.pkl",
        "term2pre_cases_valid.pkl",
        "term2pre_controls_valid.pkl",
        "term2pre_cases_test.pkl",
        "term2pre_controls_test.pkl",
    ]
    existing = [name for name in output_files if (output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output files already exist: {existing}. Pass --overwrite to replace them."
        )

    print("[Split] Loading proteomics dates...")
    proteomics_date = pd.read_csv(args.proteomics_date_csv)
    required_cols = {"EID", "proteomics_test_date"}
    if not required_cols.issubset(proteomics_date.columns):
        raise ValueError(f"{args.proteomics_date_csv} must contain columns: {sorted(required_cols)}")

    proteomics_date["EID"] = proteomics_date["EID"].map(normalize_eid)
    proteomics_date["proteomics_test_date"] = pd.to_datetime(
        proteomics_date["proteomics_test_date"], errors="coerce"
    )
    proteomics_date = proteomics_date.dropna(subset=["proteomics_test_date"])
    id2protein_date = dict(zip(proteomics_date["EID"], proteomics_date["proteomics_test_date"]))

    rng = np.random.RandomState(args.seed)
    eids = np.array(list(id2protein_date.keys()), dtype=object)
    rng.shuffle(eids)
    train_end = int(0.8 * len(eids))
    valid_end = int(0.9 * len(eids))
    train_eids = set(eids[:train_end])
    valid_eids = set(eids[train_end:valid_end])
    test_eids = set(eids[valid_end:])
    print(f"[Split] Participant split: train={len(train_eids)}, valid={len(valid_eids)}, test={len(test_eids)}")

    print("[Split] Loading disease dictionaries...")
    term2caseids = load_pickle(data_dir / "term2caseids.pkl")
    term2casedates = load_pickle(data_dir / "term2casedates.pkl")
    term2controlids = load_pickle(data_dir / "term2controlids.pkl")

    records = {}
    skipped_missing_dates = 0
    skipped_invalid_dates = 0
    skipped_terms = 0

    print("[Split] Building prevalent task records...")
    for idx, term in enumerate(term2caseids):
        if term not in term2casedates or term not in term2controlids:
            skipped_terms += 1
            continue

        pre_cases = []
        incident_cases = set()
        for eid_raw, case_date_raw in zip(term2caseids[term], term2casedates[term]):
            eid = normalize_eid(eid_raw)
            protein_date = id2protein_date.get(eid)
            if protein_date is None:
                skipped_missing_dates += 1
                continue
            try:
                case_date = to_timestamp(case_date_raw)
            except Exception:
                skipped_invalid_dates += 1
                continue
            if case_date is None:
                skipped_invalid_dates += 1
                continue
            if case_date < protein_date:
                pre_cases.append(eid)
            else:
                incident_cases.add(eid)

        if not (args.min_cases < len(pre_cases) < args.max_cases):
            continue

        train_cases = [eid for eid in pre_cases if eid in train_eids]
        valid_cases = [eid for eid in pre_cases if eid in valid_eids]
        test_cases = [eid for eid in pre_cases if eid in test_eids]

        train_controls, valid_controls, test_controls = [], [], []
        total_pre_controls = 0
        for eid_raw in term2controlids[term]:
            eid = normalize_eid(eid_raw)
            if eid in incident_cases:
                continue
            total_pre_controls += 1
            if eid in train_eids:
                train_controls.append(eid)
            elif eid in valid_eids:
                valid_controls.append(eid)
            elif eid in test_eids:
                test_controls.append(eid)

        if total_pre_controls <= args.min_cases:
            continue

        records[term] = {
            "case_set": set(pre_cases),
            "train_cases": train_cases,
            "train_controls": train_controls,
            "valid_cases": valid_cases,
            "valid_controls": valid_controls,
            "test_cases": test_cases,
            "test_controls": test_controls,
        }

        if (idx + 1) % 100 == 0:
            print(f"[Split] Processed {idx + 1} terms; valid prevalent records={len(records)}")

    valid_terms = list(records.keys())
    print(f"[Split] Valid prevalent terms: {len(valid_terms)}")
    if not valid_terms:
        raise RuntimeError("No valid prevalent terms were generated. Check input dates and EID formats.")

    term2pre_cases_train, term2pre_controls_train = {}, {}
    term2pre_cases_valid, term2pre_controls_valid = {}, {}
    term2pre_cases_test, term2pre_controls_test = {}, {}

    for term in rng.permutation(valid_terms):
        rec = records[term]
        if len(rec["test_cases"]) > args.min_cases and len(rec["test_controls"]) > len(rec["test_cases"]):
            add_to_split(term2pre_cases_test, term2pre_controls_test, term, rec, "test_cases", "test_controls")
        if len(term2pre_cases_test) > args.heldout_fraction * len(valid_terms):
            break

    for term in rng.permutation(valid_terms):
        if term in term2pre_cases_test:
            continue
        rec = records[term]
        if len(rec["valid_cases"]) > args.min_cases and len(rec["valid_controls"]) > len(rec["valid_cases"]):
            add_to_split(term2pre_cases_valid, term2pre_controls_valid, term, rec, "valid_cases", "valid_controls")
        if len(term2pre_cases_valid) > args.heldout_fraction * len(valid_terms):
            break

    for term in rng.permutation(valid_terms):
        if term in term2pre_cases_test or term in term2pre_cases_valid:
            continue
        rec = records[term]
        if len(rec["train_cases"]) > args.train_min_cases and len(rec["train_controls"]) > len(rec["train_cases"]):
            add_to_split(term2pre_cases_train, term2pre_controls_train, term, rec, "train_cases", "train_controls")

    removed_test_overlap = []
    for term in list(term2pre_cases_test):
        test_set = records[term]["case_set"]
        for other_term in list(term2pre_cases_train):
            score = jaccard(test_set, records[other_term]["case_set"])
            if score > args.jaccard_threshold:
                removed_test_overlap.append((term, "train", other_term, score))
                term2pre_cases_train.pop(other_term, None)
                term2pre_controls_train.pop(other_term, None)
        for other_term in list(term2pre_cases_valid):
            score = jaccard(test_set, records[other_term]["case_set"])
            if score > args.jaccard_threshold:
                removed_test_overlap.append((term, "valid", other_term, score))
                term2pre_cases_valid.pop(other_term, None)
                term2pre_controls_valid.pop(other_term, None)

    removed_valid_overlap = []
    for term in list(term2pre_cases_valid):
        valid_set = records[term]["case_set"]
        for other_term in list(term2pre_cases_train):
            score = jaccard(valid_set, records[other_term]["case_set"])
            if score > args.jaccard_threshold:
                removed_valid_overlap.append((term, "train", other_term, score))
                term2pre_cases_train.pop(other_term, None)
                term2pre_controls_train.pop(other_term, None)

    dump_pickle(output_dir / "term2pre_cases_train.pkl", term2pre_cases_train)
    dump_pickle(output_dir / "term2pre_controls_train.pkl", term2pre_controls_train)
    dump_pickle(output_dir / "term2pre_cases_valid.pkl", term2pre_cases_valid)
    dump_pickle(output_dir / "term2pre_controls_valid.pkl", term2pre_controls_valid)
    dump_pickle(output_dir / "term2pre_cases_test.pkl", term2pre_cases_test)
    dump_pickle(output_dir / "term2pre_controls_test.pkl", term2pre_controls_test)

    metadata = {
        "seed": args.seed,
        "min_cases": args.min_cases,
        "max_cases": args.max_cases,
        "train_min_cases": args.train_min_cases,
        "heldout_fraction": args.heldout_fraction,
        "jaccard_threshold": args.jaccard_threshold,
        "participant_split": {
            "train": len(train_eids),
            "valid": len(valid_eids),
            "test": len(test_eids),
        },
        "num_valid_prevalent_terms": len(valid_terms),
        "num_tasks": {
            "train": len(term2pre_cases_train),
            "valid": len(term2pre_cases_valid),
            "test": len(term2pre_cases_test),
        },
        "removed_overlap_with_test": len(removed_test_overlap),
        "removed_overlap_with_valid": len(removed_valid_overlap),
        "skipped_terms": skipped_terms,
        "skipped_missing_dates": skipped_missing_dates,
        "skipped_invalid_dates": skipped_invalid_dates,
    }
    with open(output_dir / "term2pre_split_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print("[Split] Finished.")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    for name in output_files:
        print(f"[Split] Wrote {output_dir / name}")


if __name__ == "__main__":
    main()
