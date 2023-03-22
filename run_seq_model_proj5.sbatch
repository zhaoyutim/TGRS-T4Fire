#!/usr/bin/env bash
#SBATCH --output /geoinfo_vol1/zhao2/run_logs/%J_slurm.out
#SBATCH --error  /geoinfo_vol1/zhao2/run_logs/%J_slurm.err
#SBATCH --mem  50GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. /geoinfo_vol1/zhao2/mambaforge/etc/profile.d/conda.sh
conda activate myenv
echo $LD_LIBRARY_PATH
PYTHONUNBUFFERED=1; python3 run_seq_model_proj5.py -m $1 -b $2 -r $3 -lr $4 -nh $5 -md $6 -ed $7 -nl $8 -mode $9
