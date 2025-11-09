#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_csv.py
--------------
Build A.csv (per-image) and B.csv (paired images) from dataset.csv.

IMPORTANT: A.csv and B.csv BOTH have exactly these 10 columns:
  subject_id,image_uid,split,sex,age,diagnosis,last_diagnosis,image_path,segm_path,latent_path

Notes:
- Timestamps (from image_path/image_uid) are used ONLY to sort/pair; they are NOT written out.
- B.csv writes pairs as TWO ROWS per pair: first row = earlier (A), second row = later (B).
- Subject grouping options let you aggregate multiple scans of the same person.

Usage (basic):
  python3 scripts/prepare/prepare_csv.py --dataset_csv ./dataset.csv --output_path ./

All pairs per grouped subject, require same split, at least 1 year gap:
  python3 scripts/prepare/prepare_csv.py --dataset_csv ./dataset.csv --output_path ./ \
    --pairing all --same_split --min_years 1 --subject_group first_token --tqdm
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Robust timestamp regex (non-digit boundaries; works with underscores etc.)
TS14 = re.compile(r"(?<!\d)([12][0-9]{3})([01][0-9])([0-3][0-9])([0-2][0-9])([0-5][0-9])([0-5][0-9])(?!\d)")
TS8  = re.compile(r"(?<!\d)([12][0-9]{3})([01][0-9])([0-3][0-9])(?!\d)")
Y4   = re.compile(r"(?<!\d)([12][0-9]{3})(?!\d)")

FIELDS10 = [
    "subject_id","image_uid","split","sex","age","diagnosis","last_diagnosis",
    "image_path","segm_path","latent_path"
]

def read_dataset_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        rows = [dict(r) for r in rdr]
    out=[]
    for r in rows:
        out.append({(k or "").strip(): (v or "").strip() for k,v in r.items()})
    return out

def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS10)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS10})

def parse_timestamp_from_texts(*texts: str) -> Optional[datetime]:
    """Prefer 14-digit, then 8-digit (00:00:00), then 4-digit year (Jan 1st)."""
    for t in texts:
        if not t: continue
        m = TS14.search(t)
        if m:
            raw = "".join(m.groups())
            try: return datetime.strptime(raw, "%Y%m%d%H%M%S")
            except Exception: pass
    for t in texts:
        if not t: continue
        m = TS8.search(t)
        if m:
            raw = "".join(m.groups())
            try: return datetime.strptime(raw, "%Y%m%d")
            except Exception: pass
    for t in texts:
        if not t: continue
        m = Y4.search(t)
        if m:
            y = m.group(1)
            try: return datetime.strptime(y + "0101000000", "%Y%m%d%H%M%S")
            except Exception: return datetime.strptime(y + "0101", "%Y%m%d")
    return None

def subject_group_key(subject_id: str, mode: str, regex_pat: Optional[re.Pattern]) -> str:
    s = (subject_id or "").strip()
    if not s: return ""
    if mode == "exact":
        return s
    if mode == "first_token":
        return s.split("_", 1)[0]
    if mode == "regex" and regex_pat is not None:
        m = regex_pat.search(s)
        return m.group(1) if m else s
    return s

def build_A(rows: List[Dict[str, str]], out_dir: Path, use_tqdm: bool=False) -> List[Dict[str, str]]:
    iterator = rows
    if use_tqdm:
        try:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(rows, desc="A.csv", ncols=80)
        except Exception:
            pass
    a_rows=[]
    for r in iterator:
        # Ensure all 10 keys exist; fill blanks if missing
        a_rows.append({k: r.get(k, "") for k in FIELDS10})
    write_csv(out_dir / "A.csv", a_rows)
    return a_rows

def build_B(rows: List[Dict[str, str]],
            out_dir: Path,
            pairing: str = "edge",
            min_days: int = 0,
            min_years: float = 0.0,
            same_split: bool = False,
            max_pairs_per_subject: int = 0,
            subject_group: str = "exact",
            subject_regex: str = "",
            use_tqdm: bool = False) -> List[Dict[str, str]]:
    """Write two rows per pair (A then B). Output columns = FIELDS10 only."""
    # Enrich with parsed dt
    enriched=[]
    for r in rows:
        dt = parse_timestamp_from_texts(r.get("image_path",""), r.get("image_uid",""))
        if dt:
            rr = {k: r.get(k, "") for k in FIELDS10}
            rr["_dt"] = dt
            enriched.append(rr)

    # Group by subject key
    regex_pat = re.compile(subject_regex) if (subject_group=="regex" and subject_regex) else None
    groups: Dict[str, List[Dict[str, str]]] = {}
    for r in enriched:
        key = subject_group_key(r.get("subject_id",""), subject_group, regex_pat)
        groups.setdefault(key, []).append(r)

    # Threshold
    td_thresh = timedelta(days=float(min_days) + float(min_years)*365.2425)

    b_rows=[]
    keys = list(groups.keys())
    if use_tqdm:
        try:
            from tqdm import tqdm  # type: ignore
            keys = list(tqdm(keys, desc="B.csv", ncols=80))
        except Exception:
            pass

    for key in keys:
        lst = groups[key]
        if len(lst) < 2:
            continue
        lst.sort(key=lambda x: x["_dt"])

        if pairing == "edge":
            A_ = lst[0]
            B_ = lst[-1]
            if same_split and (A_.get("split","") != B_.get("split","")):
                continue
            if (B_["_dt"] - A_["_dt"]) < td_thresh:
                continue
            b_rows.append({k: A_.get(k, "") for k in FIELDS10})
            b_rows.append({k: B_.get(k, "") for k in FIELDS10})
        else:  # all pairs
            count = 0
            n = len(lst)
            for i in range(n):
                for j in range(i+1, n):
                    Ai, Bj = lst[i], lst[j]
                    if same_split and (Ai.get("split","") != Bj.get("split","")):
                        continue
                    if (Bj["_dt"] - Ai["_dt"]) < td_thresh:
                        continue
                    b_rows.append({k: Ai.get(k, "") for k in FIELDS10})
                    b_rows.append({k: Bj.get(k, "") for k in FIELDS10})
                    count += 1
                    if max_pairs_per_subject>0 and count>=max_pairs_per_subject:
                        break
                if max_pairs_per_subject>0 and count>=max_pairs_per_subject:
                    break

    write_csv(out_dir / "B.csv", b_rows)
    return b_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--split_filter", default="", help="Comma list (e.g., train,valid). Empty=all.")
    ap.add_argument("--pairing", choices=["edge","all"], default="edge")
    ap.add_argument("--min_days", type=int, default=0)
    ap.add_argument("--min_years", type=float, default=0.0)
    ap.add_argument("--same_split", action="store_true")
    ap.add_argument("--max_pairs_per_subject", type=int, default=0)
    ap.add_argument("--subject_group", choices=["exact","first_token","regex"], default="exact")
    ap.add_argument("--subject_regex", default="")
    ap.add_argument("--tqdm", action="store_true")
    args = ap.parse_args()

    ds_path = Path(args.dataset_csv).resolve()
    out_dir = Path(args.output_path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_dataset_rows(ds_path)
    if args.split_filter:
        allow = {s.strip() for s in args.split_filter.split(",") if s.strip()}
        rows = [r for r in rows if r.get("split","") in allow]

    print(f"📄 dataset rows in scope: {len(rows)}")
    print(f"⚙️ pairing={args.pairing}, min_days={args.min_days}, min_years={args.min_years}, same_split={args.same_split}, "
          f"max_pairs_per_subject={args.max_pairs_per_subject}, subject_group={args.subject_group}")

    # A.csv (same schema 10 cols)
    A = build_A(rows, out_dir, use_tqdm=args.tqdm)
    print(f"✅ A.csv written ({len(A)} rows) → {out_dir/'A.csv'}")

    # B.csv (two rows per pair)
    B = build_B(rows, out_dir,
                pairing=args.pairing, min_days=args.min_days, min_years=args.min_years,
                same_split=args.same_split, max_pairs_per_subject=args.max_pairs_per_subject,
                subject_group=args.subject_group, subject_regex=args.subject_regex,
                use_tqdm=args.tqdm)
    print(f"✅ B.csv written ({len(B)} rows, i.e., {len(B)//2} pairs) → {out_dir/'B.csv'}")

if __name__ == "__main__":
    main()
