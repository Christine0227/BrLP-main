#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert source CSV:
  image_id,subject_id,study_id,series_id,image_visit,image_date,image_description,split
→ dataset.csv with fields:
  subject_id,image_uid,split,sex,age,diagnosis,last_diagnosis,image_path,segm_path,latent_path

Now with visible progress:
  - prints a progress line every --log_every rows (default 500)
  - optional tqdm bar with --tqdm (pip install tqdm)
  - optional --limit to process only first N rows for quick test
"""
import argparse, csv, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def load_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        return [r for r in rdr]

def load_meta(meta_path: Optional[Path], age_is_years: bool) -> Dict[str, Dict]:
    meta = {}
    if not meta_path:
        return meta
    import pandas as pd
    df = pd.read_csv(meta_path)
    df.columns = [c.strip().lower() for c in df.columns]
    for _, r in df.iterrows():
        sid = str(r.get("subject_id", "")).strip()
        if not sid or sid == "nan":
            continue
        sex = r.get("sex", "")
        age = r.get("age", "")
        try:
            age = float(age)
            if age_is_years:
                age = age / 100.0
        except Exception:
            age = ""
        meta[sid] = {
            "sex": sex,
            "age": age,
            "diagnosis": r.get("diagnosis", ""),
            "last_diagnosis": r.get("last_diagnosis", ""),
        }
    return meta

def list_subject_dirs(pre_root: Path) -> List[Path]:
    return sorted([p for p in pre_root.iterdir() if p.is_dir() and p.name != "_ALL"])

def list_warped_under(dirpath: Path) -> List[Path]:
    cands = []
    for n in ("turboprep_Warped.nii.gz", "turboprep_Warped.nii"):
        p = dirpath / n
        if p.exists():
            cands.append(p)
    for sub in sorted([p for p in dirpath.iterdir() if p.is_dir()]):
        for n in ("turboprep_Warped.nii.gz", "turboprep_Warped.nii"):
            p = sub / n
            if p.exists():
                cands.append(p)
    return cands

def find_segm_near(warped: Path) -> Optional[Path]:
    parent = warped.parent
    for n in ("segm.nii.gz", "segm.nii"):
        p = parent / n
        if p.exists():
            return p
    return None

def normalize(s: str) -> str:
    return (s or "").strip()

def score_candidate(path: Path, subject_id: str, image_id: str, series_id: str, image_visit: str) -> Tuple[int, int]:
    s_path = path.as_posix()
    score = 0
    prefix_len = 0
    subj = normalize(subject_id); img = normalize(image_id)
    ser = normalize(series_id); vis = normalize(image_visit)

    if subj:
        for part in path.parts:
            if part == subj:
                score += 100; break
    if img:
        for part in path.parts:
            if part == img:
                score += 80; break
        for part in path.parts:
            if part.startswith(img):
                score += 60; prefix_len = max(prefix_len, len(img)); break
        if img in s_path:
            score += 40
    if ser and ser in s_path: score += 10
    if vis and vis in s_path: score += 10
    return score, prefix_len

def choose_best(cands: List[Path], subject_id: str, image_id: str, series_id: str, image_visit: str) -> Optional[Path]:
    if not cands: return None
    best = None; best_score = (-1, -1)
    for c in cands:
        sc = score_candidate(c, subject_id, image_id, series_id, image_visit)
        if sc > best_score:
            best_score = sc; best = c
    if len(cands) > 1 and best_score[0] <= 0:  # if nothing scored, reject when ambiguous
        return None
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_csv", required=True)
    ap.add_argument("--preprocessed_dir", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--metadata_csv", default="")
    ap.add_argument("--age_is_years", action="store_true")
    ap.add_argument("--log_every", type=int, default=500, help="print progress every N rows")
    ap.add_argument("--limit", type=int, default=0, help="only process first N rows (0=all)")
    ap.add_argument("--tqdm", action="store_true", help="show tqdm progress bar (pip install tqdm)")
    args = ap.parse_args()

    pre_root = Path(args.preprocessed_dir).resolve()
    if not pre_root.exists():
        print(f"❌ preprocessed_dir not found: {pre_root}", file=sys.stderr); sys.exit(1)

    rows_in = load_rows(Path(args.source_csv).resolve())
    if args.limit and args.limit > 0:
        rows_in = rows_in[:args.limit]

    meta = load_meta(Path(args.metadata_csv).resolve() if args.metadata_csv else None,
                     age_is_years=args.age_is_years)

    subj_dirs = list_subject_dirs(pre_root)
    subj2warped: Dict[str, List[Path]] = {d.name: list_warped_under(d) for d in subj_dirs}
    total_warped = sum(len(v) for v in subj2warped.values())

    print(f"📁 subjects scanned: {len(subj2warped)} | warped files indexed: {total_warped} | rows to process: {len(rows_in)}", flush=True)

    out_rows = []; skipped = 0; matched = 0
    iterator = rows_in

    if args.tqdm:
        try:
            from tqdm import tqdm
            iterator = tqdm(rows_in, ncols=80)
        except Exception:
            print("ℹ️ tqdm not installed; continuing without bar. (pip install tqdm)", flush=True)

    for idx, r in enumerate(iterator, 1):
        subject_id = normalize(r.get("subject_id"))
        image_id   = normalize(r.get("image_id"))
        series_id  = normalize(r.get("series_id"))
        image_visit= normalize(r.get("image_visit"))
        split      = normalize(r.get("split")) or "train"

        # collect candidates
        cands: List[Path] = []
        if subject_id and subject_id in subj2warped:
            cands.extend(subj2warped[subject_id])
        if image_id:
            for dname, warped_list in subj2warped.items():
                if dname.startswith(image_id) or image_id in dname:
                    cands.extend(warped_list)
        if not cands:  # fallback to all
            for warped_list in subj2warped.values():
                cands.extend(warped_list)

        warped = choose_best(cands, subject_id, image_id, series_id, image_visit)
        if not warped:
            skipped += 1
        else:
            segm = find_segm_near(warped)
            image_path = str(warped.resolve())
            segm_path = str(segm.resolve()) if segm else ""
            latent_path = image_path.replace(".nii.gz", "_latent.npz").replace(".nii", "_latent.npz")
            image_uid = image_id if image_id else (warped.name[:-7] if warped.name.endswith(".nii.gz") else warped.stem)

            sex = age = diagnosis = last_diagnosis = ""
            if subject_id in meta:
                m = meta[subject_id]
                sex = m.get("sex",""); age = m.get("age","")
                diagnosis = m.get("diagnosis",""); last_diagnosis = m.get("last_diagnosis","")

            out_rows.append({
                "subject_id": subject_id,
                "image_uid": image_uid,
                "split": split,
                "sex": sex,
                "age": age,
                "diagnosis": diagnosis,
                "last_diagnosis": last_diagnosis,
                "image_path": image_path,
                "segm_path": segm_path,
                "latent_path": latent_path,
            })
            matched += 1

        if not args.tqdm and args.log_every > 0 and idx % args.log_every == 0:
            print(f"… processed {idx}/{len(rows_in)} | matched={matched} | skipped={skipped}", flush=True)

    out_path = Path(args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject_id","image_uid","split","sex","age",
                  "diagnosis","last_diagnosis","image_path","segm_path","latent_path"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(out_rows)

    print(f"✅ dataset.csv written: {out_path} (rows: {len(out_rows)})", flush=True)
    print(f"⚠️ skipped: {skipped}", flush=True)

if __name__ == "__main__":
    main()
