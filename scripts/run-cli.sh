#!/bin/bash
#SBATCH --job-name=snakemake
#SBATCH --output=.cmmvae/logs/snakemake/job.%j.out
#SBATCH --error=.cmmvae/logs/snakemake/job.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=350GB

scripts/run-command-n-env.sh cmmvae workflow cli "$@"
