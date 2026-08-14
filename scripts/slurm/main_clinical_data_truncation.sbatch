#!/bin/bash
#SBATCH --job-name=theo-trunc-old
#SBATCH -p normalprio
#SBATCH --array=0-99%100
#SBATCH --cpus-per-task=1
#SBATCH --exclude=node17,node8,node4,node2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/clusterdata/%u/%x_%A_%a.out
#SBATCH --error=/mnt/clusterdata/%u/%x_%A_%a.err

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"

cd "${SLURM_SUBMIT_DIR}"

export PATH="/mnt/clusterdata/software/python/3.10/bin:${PATH}"
export PIPENV_VENV_IN_PROJECT=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SCRATCH:-/tmp}/${USER}/theo_trunc_mpl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

RESULTS_ROOT="${THEOPHYLLINE_RESULTS_ROOT:-results/theo/truncation_array_${SLURM_ARRAY_JOB_ID}}"
printf -v TASK_DIR "run_%03d" "${SLURM_ARRAY_TASK_ID}"
export THEOPHYLLINE_BASE_DIR="${RESULTS_ROOT}/${TASK_DIR}"

mkdir -p "${MPLCONFIGDIR}" "${THEOPHYLLINE_BASE_DIR}"

echo "START $(date)"
echo "HOST $(hostname)"
echo "TASK ${SLURM_ARRAY_TASK_ID}"
echo "RESULTS ${THEOPHYLLINE_BASE_DIR}"

pipenv run python3.10 -u scripts/main_clinical_data_truncation.py

echo "END $(date)"
