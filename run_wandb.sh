#!/bin/bash
#SBATCH --job-name=wandb_sweep              # Name your job
#SBATCH --output=logs/sweep_%A_%a.out       # .out file to capture outputs
#SBATCH --error=logs/sweep_%A_%a.err        # .err files to capture error
#SBATCH --qos=debug                         # NERSC docs provide a helpful flowchart for this
#SBATCH --constraint=cpu                    # Perlmutter requires you to specify cpu or gpu mode
#SBATCH --account=nguest                    # NERSC project associated with this job
#SBATCH --nodes=1                           # Book jobs on the same node
#SBATCH --ntasks-per-node=1                 # Select 1 task/node and wandb will spawn
#SBATCH --cpus-per-task=32                  # Match n_jobs=-1 in train.py
#SBATCH --time=00:30:00                     # Run time in HH:MM:SS
#SBATCH --mail-type=BEGIN,END,FAIL          # Receive emails with job status
#SBATCH --mail-user=ellianna@berkeley.edu   

# Create log directory
mkdir -p logs

# Add the CVMFS path to your path to access apptainer
export PATH=${PATH}:/cvmfs/oasis.opensciencegrid.org/mis/apptainer/1.3.3/x86_64/bin

# Replace with the sweep ID that you saved
SWEEP_ID="username/my-sweep-name/sweepID"
CONTAINER="wandb_latest.sif"

# Redirect W&B local logs to /tmp to avoid error
export WANDB_DIR=/tmp

# Run the W&B agent inside the container
apptainer exec $CONTAINER \
    wandb agent $SWEEP_ID
