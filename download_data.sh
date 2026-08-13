#!/usr/bin/env bash
# Downloads the datasets and PSMAP surrogate files needed by both pipelines
# in this repo (non_phase_shear/ and phase_shear/), from the Google Drive
# backup of the parent repo's data/ and output-files/ directories.
#
# Requires: rclone (https://rclone.org/downloads/), configured with a remote
# named "gdrive" that has access to
#   gdrive:PhD/data/gaussian-wavefront-spatially-resolved-inference/{data,output-files}
# Run `rclone config` first if you don't already have this remote set up.
#
# What gets fetched (~31 GB total -- the remote has other datasets/files
# too; this only pulls what the scripts in this repo actually read):
#   - output-files/PSGRID4D_CONFOCAL_FINE_{Z0,Z100}.h5  (~3 GB)
#       The point-spread-map surrogates. Required for BOTH pipelines, and
#       for generating brand new datasets with generate_data.py -- this is
#       the one prerequisite you can't skip.
#   - data/R80_N200_A1000000..._f0.3000/       (1e6 atoms, no phase ramp, ~9.4 GB)
#   - data/R40_N50_A100000000..._f0.3000/      (1e8 atoms, no phase ramp, ~5.0 GB)
#       Used by non_phase_shear/ (null / MAP-moments / pixel-likelihood / oracle).
#   - data/R20_N200_A1000000..._kappa3.14e+04_both/    (1e6 atoms, phase ramp, ~2.4 GB)
#   - data/R20_N200_A100000000..._kappa3.14e+04_both/  (1e8 atoms, phase ramp, ~11 GB)
#       Used by phase_shear/ (raw / fitted-feature regression / oracle regression).
#
# All four datasets can alternatively be regenerated from scratch with
# generate_data.py once the PSMAP files are in place -- see README.md.
set -euo pipefail
cd "$(dirname "$0")"

REMOTE="gdrive:PhD/data/gaussian-wavefront-spatially-resolved-inference"

DATASET_1E6="R80_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000"
DATASET_1E8="R40_N50_A100000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000"
DATASET_1E6_SHEAR="R20_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000_kappa3.14e+04_both"
DATASET_1E8_SHEAR="R20_N200_A100000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000_kappa3.14e+04_both"

echo "== Downloading fine PSMAP surrogates (~3 GB, required for everything) =="
mkdir -p output-files
rclone copy "$REMOTE/output-files/PSGRID4D_CONFOCAL_FINE_Z0.h5" output-files/ --progress
rclone copy "$REMOTE/output-files/PSGRID4D_CONFOCAL_FINE_Z100.h5" output-files/ --progress

echo "== Downloading non_phase_shear datasets (~14.4 GB) =="
rclone copy "$REMOTE/data/$DATASET_1E6" "data/$DATASET_1E6" --progress
rclone copy "$REMOTE/data/$DATASET_1E8" "data/$DATASET_1E8" --progress

echo "== Downloading phase_shear datasets (~13.4 GB) =="
rclone copy "$REMOTE/data/$DATASET_1E6_SHEAR" "data/$DATASET_1E6_SHEAR" --progress
rclone copy "$REMOTE/data/$DATASET_1E8_SHEAR" "data/$DATASET_1E8_SHEAR" --progress

echo "Done. data/ and output-files/ are populated."
