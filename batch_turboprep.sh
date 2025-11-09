#!/bin/bash
# Batch turboprep for multiple NIfTI inputs on WSL (resume-safe + final aggregation)
# Usage:
#   chmod +x batch_turboprep.sh
#   ./batch_turboprep.sh
set -euo pipefail

########################
# === USER PATHS ===
########################
INPUT="/mnt/c/Users/CPS/Desktop/BrLP-main/nii"
OUTPUT="/mnt/c/Users/CPS/Desktop/BrLP-main/preprocessed"
TEMPLATE="/mnt/c/Users/CPS/Desktop/BrLP-main/MNI152_T1_1mm_brain.nii.gz"

########################
# === PARAMETERS ===
########################
IMAGE="docker.io/lemuelpuglisi/turboprep:latest"
MODALITY="-m t1"    # change to "-m flair" or "-m t2" if needed
KEEP="--keep"       # keep intermediates for debugging

########################
# === PRECHECKS ===
########################
# 0) Docker check
if ! command -v docker >/dev/null 2>&1; then
  echo "❌ Docker not found. Install Docker Desktop and enable WSL integration."
  exit 1
fi

mkdir -p "$OUTPUT"

# 1) Template check (auto-download if missing)
if [[ ! -f "$TEMPLATE" ]]; then
  echo "🔎 Template not found, downloading MNI152_T1_1mm_brain.nii.gz ..."
  mkdir -p "$(dirname "$TEMPLATE")"
  curl -fL \
    "https://raw.githubusercontent.com/Washington-University/HCPpipelines/master/global/templates/MNI152_T1_1mm_brain.nii.gz" \
    -o "$TEMPLATE"
  echo "✅ Template saved: $TEMPLATE"
fi

# 2) Collect inputs
mapfile -t FILES < <(find "$INPUT" -maxdepth 1 -type f \( -iname '*.nii' -o -iname '*.nii.gz' \) | sort)
if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "⚠️  No NIfTI files found in: $INPUT"
  exit 1
fi

# 3) Pull container (quiet)
echo "🐳 Pulling image: $IMAGE"
docker pull "$IMAGE" >/dev/null

echo "===== turboprep batch (RESUME ENABLED) ====="
echo "IN : $INPUT"
echo "OUT: $OUTPUT"
echo "TPL: $TEMPLATE"
echo "IMG: $IMAGE"
echo "MOD: $MODALITY"
echo "============================================"

########################
# === PROCESS LOOP ===
########################
for f in "${FILES[@]}"; do
  name="$(basename "$f")"
  base="${name%.nii.gz}"; base="${base%.nii}"
  outdir="$OUTPUT/$base"

  # resume: skip if output folder exists and not empty
  if [[ -d "$outdir" && -n "$(ls -A "$outdir" 2>/dev/null)" ]]; then
    echo "⏭️  Skip (already processed): $base"
    continue
  fi

  mkdir -p "$outdir"
  echo "▶ Processing: $name → $outdir"

  docker run --rm \
    -v "$INPUT":/in:ro \
    -v "$OUTPUT":/out \
    -v "$TEMPLATE":/tmpl.nii.gz:ro \
    "$IMAGE" \
    "/in/$name" "/out/$base" "/tmpl.nii.gz" $MODALITY $KEEP

  echo "✅ Done: $base"
done

########################
# === AGGREGATION ===
########################
ALLDIR="$OUTPUT/_ALL"
ALIGNDIR="$ALLDIR/aligned"
MANIFEST="$ALLDIR/manifest.csv"
mkdir -p "$ALIGNDIR"

# header
echo "subject,warped_path,seg_path,mask_path" > "$MANIFEST"

count=0
while IFS= read -r -d '' subdir; do
  subj="$(basename "$subdir")"
  # common filenames from turboprep
  warped="$subdir/turboprep_Warped.nii.gz"
  [[ -f "$warped" ]] || warped="$subdir/turboprep_Warped.nii"
  seg="$subdir/segm.nii.gz";    [[ -f "$seg"  ]] || seg="$subdir/segm.nii"
  mask="$subdir/brain_mask.nii.gz"; [[ -f "$mask" ]] || mask="$subdir/brain_mask.nii"

  if [[ -f "$warped" ]]; then
    mkdir -p "$ALIGNDIR"
    # ensure gz output
    if [[ "$warped" == *.nii.gz ]]; then
      cp -f "$warped" "$ALIGNDIR/${subj}.nii.gz"
    else
      gzip -c "$warped" > "$ALIGNDIR/${subj}.nii.gz"
    fi
    echo "${subj},$warped,${seg:-},${mask:-}" >> "$MANIFEST"
    ((count++))
  else
    echo "⚠️  Missing turboprep_Warped for $subj; skip collect."
  fi
done < <(find "$OUTPUT" -mindepth 1 -maxdepth 1 -type d ! -name "_ALL" -print0)

echo "📄 Manifest: $MANIFEST"
echo "📦 Collected aligned volumes: $ALIGNDIR (total: $count)"

########################
# === 4D & MEAN ===
########################
if [[ $count -gt 0 ]]; then
  echo "🧠 Building 4D stack and mean image (Docker Python)..."
  docker run --rm -v "$ALLDIR":/data python:3.11-slim bash -lc "
    pip -q install nibabel numpy >/dev/null && python - <<'PY'
import os, glob, nibabel as nib, numpy as np
align_dir = '/data/aligned'
out4d = '/data/all_aligned_4d.nii.gz'
outmean = '/data/all_aligned_mean.nii.gz'
files = sorted(glob.glob(os.path.join(align_dir, '*.nii')) + glob.glob(os.path.join(align_dir, '*.nii.gz')))
if not files:
    raise SystemExit('no files')
imgs = [nib.load(f) for f in files]
shapes = {img.shape for img in imgs}
if len(shapes)!=1:
    raise RuntimeError(f'Shape mismatch: {shapes}')
data = np.stack([img.get_fdata(dtype=np.float32) for img in imgs], axis=-1)
ref = imgs[0]
nib.save(nib.Nifti1Image(data, ref.affine, ref.header), out4d)
nib.save(nib.Nifti1Image(data.mean(axis=-1), ref.affine, ref.header), outmean)
print(f'Stacked {len(files)} → {out4d}\\nMean → {outmean}')
PY"
  echo "✅ 4D & mean saved in $ALLDIR"
else
  echo "ℹ️  No aligned volumes collected; skip 4D/mean."
fi

echo "🎉 All done."
