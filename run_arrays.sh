#!/bin/bash
#SBATCH --job-name=array_intro              # Name your job
#SBATCH --output=logs/array_%A_%a.out       # .out file to capture outputs
#SBATCH --error=logs/array_%A_%a.err        # .err files to capture error
#SBATCH --qos=debug                         # NERSC docs provide a helpful flowchart
#SBATCH --constraint=cpu                    # Perlmutter requires cpu or gpu spec
#SBATCH --account=your-project      		    # NERSC project associated with this job
#SBATCH --time=00:05:00                     # Run time in HH:MM:SS
#SBATCH --mail-type=BEGIN,END,FAIL          # Receive emails with job status
#SBATCH --mail-user=your-email@email.gov

# Create log directory
mkdir -p logs

# Hello World from each array task
echo "Hello World! From array task ${SLURM_ARRAY_TASK_ID} in job ${SLURM_JOB_ID} on $(hostname)"
sleep 10
