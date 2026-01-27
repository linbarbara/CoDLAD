#!/usr/bin/env bash
set -euo pipefail

# Default behavior matches the repository code:
# - TCGA downstream only (see classifier.py main_train_classifier)
# - i runs from 0 to RUNS-1
# - output folders follow ./result/... conventions

RUNS=1
RESULT_ROOT="./result"
PRETRAIN_SUBDIR="pretrain_vae_latent_diffusion"
CLASSIFIER_SUBDIR="classifier_vae_latent_diffusion"

# Optional: choose python executable
PYTHON_BIN="python"

usage() {
  echo "Usage: $0 [-n RUNS] [-r RESULT_ROOT] [-p PYTHON_BIN]" | cat
  echo "  -n RUNS         number of repetitions (default: 1)" | cat
  echo "  -r RESULT_ROOT  root folder for outputs (default: ./result)" | cat
  echo "  -p PYTHON_BIN   python executable (default: python)" | cat
}

while getopts ":n:r:p:h" opt; do
  case "$opt" in
    n) RUNS="$OPTARG" ;;
    r) RESULT_ROOT="$OPTARG" ;;
    p) PYTHON_BIN="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) usage; exit 1 ;;
  esac
done

for ((i=0; i<RUNS; i++)); do
  PRETRAIN_OUTFOLDER="${RESULT_ROOT}/${PRETRAIN_SUBDIR}/pretrain${i}"
  CLASSIFIER_OUTFOLDER="${RESULT_ROOT}/${CLASSIFIER_SUBDIR}"

  echo "[Run ${i}] Pretrain -> ${PRETRAIN_OUTFOLDER}"

  # Default pretraining (uses built-in CCLE/TCGA pretrain files referenced in data.py: pretrain_data())
  ${PYTHON_BIN} -u pretrain.py --outfolder "${PRETRAIN_OUTFOLDER}"

  echo "[Run ${i}] Classifier (TCGA default in classifier.py) -> ${CLASSIFIER_OUTFOLDER}"

  # Default classifier run (TCGA) using the pretrained folder
  ${PYTHON_BIN} -u classifier.py \
    --dataset TCGA \
    --pretrain_model "${PRETRAIN_OUTFOLDER}" \
    --outfolder "${CLASSIFIER_OUTFOLDER}" \
    --outname "vae_latent_result${i}.csv"
  # ---------------------------------------------------------------------------
  # OPTIONAL (commented): external pretraining input
  # If you want to pretrain on your own source/target gene expression matrices,
  # you must also provide tissue label CSVs for both domains.
  #
  # ${PYTHON_BIN} pretrain.py \
  #   --outfolder "${PRETRAIN_OUTFOLDER}" \
  #   --source /path/to/source_expression.csv \
  #   --target /path/to/target_expression.csv \
  #   --source_tissue /path/to/source_tissue.csv \
  #   --target_tissue /path/to/target_tissue.csv
  # ---------------------------------------------------------------------------

  # ---------------------------------------------------------------------------
  # OPTIONAL (commented): run PDTC
  # ${PYTHON_BIN} classifier.py \
  #   --dataset PDTC \
  #   --pretrain_model "${PRETRAIN_OUTFOLDER}" \
  #   --outfolder "${CLASSIFIER_OUTFOLDER}" \
  #   --outname "pdtc_result${i}.csv"
  # ---------------------------------------------------------------------------

  # ---------------------------------------------------------------------------
  # OPTIONAL (commented): run your own downstream dataset ("other")
  # Requires:
  # - --data: folder containing per-drug subfolders with sourcedata.csv/targetdata.csv/sourcelabel.csv/targetlabel.csv
  # - --drug: CSV with drug names and smiles
  #
  # ${PYTHON_BIN} classifier.py \
  #   --dataset other \
  #   --data /path/to/your_data_folder \
  #   --drug /path/to/your_drug_list.csv \
  #   --pretrain_model "${PRETRAIN_OUTFOLDER}" \
  #   --outfolder "${CLASSIFIER_OUTFOLDER}" \
  #   --outname "other_result${i}.csv"
  # ---------------------------------------------------------------------------

done

