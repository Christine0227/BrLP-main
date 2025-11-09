#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_csv_build.py (with tqdm)
--------------------------------
Builds dataset.csv from ADNI + manifest CSVs + preprocessed MRI directory.

Encodings applied:
- sex: Male -> 0, Female -> 1
- age: normalized to age/100 (three decimals)
- diagnosis: CN -> 0, MCI -> 0.5, AD -> 1
"""

from __future__ import annotations
import argparse, csv, re, random
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm

ts_patterns = [
    re.compile(r"(?<!\d)([12][0-9]{3})([01][0-9])([0-3][0-9])([0-2][0-9])([0-5][0-9])([0-5][0-9])(?!\d)"),
    re.compile(r"(?<!\d)([12][0-9]{3})([01][0-9])([0-3][0-9])(?!\d)")
]
FIELDS10 = ["subject_id","image_uid","split","sex","age","diagnosis","last_diagnosis","image_path","segm_path","latent_path"]

def read_rows(path:Path)->List[Dict[str,str]]:
    with path.open('r', newline='') as f:
        rdr = csv.DictReader(f)
        return [{(k or '').strip():(v or '').strip() for k,v in r.items()} for r in rdr]

def write_csv(path:Path, rows:List[Dict[str,str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS10)
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k,'') for k in FIELDS10})

def parse_date_from_values(*vals:str)->Optional[datetime]:
    for v in vals:
        if not v: continue
        for fmt in ("%Y-%m-%d","%m/%d/%Y","%Y/%m/%d","%d-%b-%Y","%d/%m/%Y"):
            try: return datetime.strptime(v,fmt)
            except: pass
        for rx in ts_patterns:
            m = rx.search(v)
            if m:
                s = ''.join(m.groups())
                for fmt in ("%Y%m%d%H%M%S","%Y%m%d"):
                    try: return datetime.strptime(s,fmt)
                    except: pass
    return None

def pick_first(*vals:str)->str:
    for v in vals:
        if v: return v
    return ''

def normalize_sex(v:str)->str:
    v = (v or '').strip().upper()
    if v in {"M","MALE"}: return "M"
    if v in {"F","FEMALE"}: return "F"
    return v

def normalize_dx(v:str)->str:
    v = (v or '').strip().upper()
    if v in {"CN","CONTROL","NORMAL"}: return "0"
    if v in {"MCI","EMCI","LMCI"}:     return "0.5"
    if v in {"AD","DEMENTIA","ALZHEIMERS"}: return "1"
    return ""

class ColFinder:
    def __init__(self, headers:List[str]):
        canon = {}
        for h in headers:
            k = re.sub(r"[^a-z0-9]","",(h or '').lower())
            if k and k not in canon:
                canon[k]=h
        self.canon=canon
    def find(self,*patterns:str)->Optional[str]:
        for pat in patterns:
            rx=re.compile(pat)
            for k,orig in self.canon.items():
                if rx.fullmatch(k) or rx.search(k):
                    return orig
        return None

def build_dataset(adni_rows, manifest_rows, pre_dir:Optional[Path], auto_split, seed):
    random.seed(seed)
    adni_f = ColFinder(list(adni_rows[0].keys()) if adni_rows else [])
    man_f  = ColFinder(list(manifest_rows[0].keys()) if manifest_rows else [])

    adni_uid_col = adni_f.find(r"^imageuid$",r"image(data)?id",r"imageuid")
    man_uid_col  = man_f.find(r"^imageuid$",r"image(data)?id",r"imageuid",r"seriesid")
    adni_subj_col= adni_f.find(r"^ptid$",r"subject(id)?$",r"rid",r"subject_?key$")
    man_subj_col = man_f.find(r"^ptid$",r"subject(id)?$",r"rid",r"subject_?key$",r"participant")
    adni_date_col= adni_f.find(r"exam(date)?$",r"scan(date)?$",r"imagedate$",r"date$")
    man_date_col = man_f.find(r"exam(date)?$",r"scan(date)?$",r"imagedate$",r"date$",r"acq(date)?$")
    sex_col      = adni_f.find(r"ptgender$",r"sex$",r"gender$")
    age_col      = adni_f.find(r"age(_?atscan)?$",r"age$",r"ageyears$")
    dx_col       = adni_f.find(r"dx(change)?$",r"dx$",r"diagnosis$",r"dx_bl$")
    split_col    = pick_first(man_f.find(r"split$") or '', adni_f.find(r"split$") or '') or None

    idx_uid, idx_subj = {}, {}
    for r in adni_rows:
        uid_raw = (r.get(adni_uid_col,'') if adni_uid_col else '').strip()
        uid = re.sub(r'^[iI]','',uid_raw)
        if uid: idx_uid[uid]=r
        subj = (r.get(adni_subj_col,'') if adni_subj_col else '').strip()
        dt = parse_date_from_values(r.get(adni_date_col,'') if adni_date_col else '')
        if subj: idx_subj.setdefault(subj,[]).append((dt,r))
    for k in idx_subj: idx_subj[k].sort(key=lambda x:(x[0]is None,x[0]))

    def find_adni(mr):
        uid=(mr.get(man_uid_col,'')if man_uid_col else '').strip()
        if uid and uid in idx_uid: return idx_uid[uid]
        subj=(mr.get(man_subj_col,'')if man_subj_col else '').strip()
        if not subj: return None
        candidates=idx_subj.get(subj,[])
        if not candidates: return None
        mdt=parse_date_from_values(mr.get(man_date_col,'')if man_date_col else '')
        if mdt is None: return candidates[-1][1]
        best,best_abs=None,None
        for dt,rr in candidates:
            if dt is None: continue
            diff=abs((dt-mdt).total_seconds())
            if best_abs is None or diff<best_abs: best_abs=diff; best=rr
        return best if best else candidates[-1][1]

    # Pre-index pre_dir
    path_map={}
    if pre_dir and pre_dir.exists():
        for p in tqdm(list(pre_dir.rglob('*Warped.nii.gz')),desc='Indexing pre_dir',ncols=80):
            m=re.search(r'I?(\d+)(?=_)',p.parent.name)
            if m: uid=m.group(1); path_map.setdefault(uid,{})['img']=str(p)
        for p in tqdm(list(pre_dir.rglob('segm.nii.gz')),desc='Indexing segm',ncols=80):
            m=re.search(r'I?(\d+)(?=_)',p.parent.name)
            if m: uid=m.group(1); path_map.setdefault(uid,{})['segm']=str(p)

    out_rows=[]
    for mr in tqdm(manifest_rows,desc='Merging manifest',ncols=80):
        ad=find_adni(mr)or{}
        subj=pick_first(mr.get(man_subj_col,''),ad.get(adni_subj_col,''))
        uid_raw=pick_first(mr.get(man_uid_col,''),ad.get(adni_uid_col,''))
        uid=re.sub(r'^[iI]','',uid_raw)
        if not subj and not uid: continue

        # ---- Encode features ----
        sex_raw=normalize_sex(pick_first(ad.get(sex_col,'')))
        sex="0" if sex_raw=="M" else "1" if sex_raw=="F" else ""

        age_raw=pick_first(ad.get(age_col,''))
        try: age_val=float(age_raw) if age_raw else None
        except ValueError: age_val=None
        age=f"{age_val/100:.3f}" if age_val is not None else ""

        dx_raw=pick_first(ad.get(dx_col,''))
        dx=normalize_dx(dx_raw)
        # --------------------------

        split=mr.get(split_col,'') if (split_col and split_col in mr) else ad.get(split_col,'') if (split_col and split_col in ad) else ''
        img_path=path_map.get(uid,{}).get('img','')
        segm_path=path_map.get(uid,{}).get('segm','')

        out_rows.append({
            'subject_id':subj,
            'image_uid':uid,
            'split':split,
            'sex':sex,
            'age':age,
            'diagnosis':dx,
            'last_diagnosis':dx,
            'image_path':img_path,
            'segm_path':segm_path,
            'latent_path':''
        })

    if auto_split:
        by_subj={}
        for i,r in enumerate(out_rows): by_subj.setdefault(r['subject_id'],[]).append(i)
        for subj,idxs in by_subj.items():
            random.shuffle(idxs)
            n=len(idxs); n_tr=int(0.8*n); n_val=int(0.1*n)
            for j,ii in enumerate(idxs):
                sp='train' if j<n_tr else 'val' if j<n_tr+n_val else 'test'
                if not out_rows[ii]['split']: out_rows[ii]['split']=sp
    return out_rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--adni_csv',required=True)
    ap.add_argument('--manifest_csv',required=True)
    ap.add_argument('--preprocessed_dir',default='')
    ap.add_argument('--output_path',required=True)
    ap.add_argument('--auto_split',action='store_true')
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()

    adni_path=Path(args.adni_csv).resolve()
    man_path=Path(args.manifest_csv).resolve()
    out_dir=Path(args.output_path).resolve()
    pre_dir=Path(args.preprocessed_dir).resolve() if args.preprocessed_dir else None

    adni_rows=read_rows(adni_path)
    man_rows=read_rows(man_path)
    print(f"ℹ️ ADNI rows: {len(adni_rows)} | manifest rows: {len(man_rows)}")

    out_rows=build_dataset(adni_rows,man_rows,pre_dir,args.auto_split,args.seed)

    ds_path=out_dir/'dataset.csv'
    write_csv(ds_path,out_rows)
    print(f"✅ dataset.csv written ({len(out_rows)} rows) → {ds_path}")
    print("Next: run prepare_csv.py to build A.csv / B.csv")

if __name__=='__main__':
    main()
