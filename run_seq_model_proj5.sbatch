#!/usr/bin/env bash
#SBATCH --output /geoinfo_vol1/zhao2/run_logs/%J_slurm.out
#SBATCH --error  /geoinfo_vol1/zhao2/run_logs/%J_slurm.err
#SBATCH --mem  80GB
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 4

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. /geoinfo_vol1/zhao2/mambaforge/etc/profile.d/conda.sh
conda activate myenv
PYTHONUNBUFFERED=1; python3 run_seq_model_proj5.py -m $1 -w $2 -p $3 -b $4 -r $5 -lr $6 -nh $7 -md $8 -ed $9 -nl ${10} -mode ${11}
