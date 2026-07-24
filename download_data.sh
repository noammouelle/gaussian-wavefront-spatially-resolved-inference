#!/usr/bin/env bash
# Downloads the two simulated datasets (1e6 atoms / 1e8 atoms) and the two
# fine-grid PSMAP surrogate files actually used by the analysis scripts in
# this branch, from the Google Drive backup made of the full
# gaussian-wavefront-spatially-resolved-inference repo's data/ and
# output-files/ directories.
#
# Requires: rclone (https://rclone.org/downloads/), configured with a remote
# named "gdrive" that has access to
#   gdrive:PhD/data/gaussian-wavefront-spatially-resolved-inference/{data,output-files}
# Run `rclone config` first if you don't already have this remote set up
# (needs at least drive.file scope / access to the specific shared folder).
#
# Total download size: ~14.4 GB (data/) + ~3 GB (output-files/, 2 files
# only) = ~17.4 GB. The remote directories contain other datasets/files not
# needed here; this script fetches only the two dataset trees and two PSMAP
# files this branch's scripts actually read.
set -euo pipefail
cd "$(dirname "$0")"

REMOTE="gdrive:PhD/data/gaussian-wavefront-spatially-resolved-inference"

DATASET_1E6="R80_N200_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000"
DATASET_1E8="R40_N50_A100000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx100um_sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000"

echo "== Downloading 1e6-atom dataset (~9.4 GB) =="
rclone copy "$REMOTE/data/$DATASET_1E6" "data/$DATASET_1E6" --progress

echo "== Downloading 1e8-atom dataset (~5.0 GB) =="
rclone copy "$REMOTE/data/$DATASET_1E8" "data/$DATASET_1E8" --progress

echo "== Downloading fine PSMAP surrogates (~3 GB) =="
mkdir -p output-files
rclone copy "$REMOTE/output-files/PSGRID4D_CONFOCAL_FINE_Z0.h5" output-files/ --progress
rclone copy "$REMOTE/output-files/PSGRID4D_CONFOCAL_FINE_Z100.h5" output-files/ --progress

echo "Done. data/ and output-files/ are populated."
