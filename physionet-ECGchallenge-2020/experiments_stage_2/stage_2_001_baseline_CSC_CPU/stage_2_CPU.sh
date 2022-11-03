#!/bin/bash
#SBATCH --job-name=s2_001_cpu
#SBATCH --account=Project_2002658
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user=joaquin.j.rives@utu.fi

# note, this job requests a total of 1 cores and 1 GPGPU cards
# note, submit the job from taito-gpu.csc.fi
# commands to manage the batch script
#   submission command
#     sbatch [script-file]
#   status command
#     squeue -u jafarita
#   termination command
#     scancel [jobid]


# example run commands

module load tensorflow/1.14.0

srun python3 stage_2.py

# This script will print some usage statistics to the
# end of the standard out file
# Use that to improve your resource request estimate
# on later jobs.
seff $SLURM_JOBID
