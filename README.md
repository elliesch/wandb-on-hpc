# Running Hyperparameter Sweeps on Distributed Systems with Weights & Biases

In this tutorial, you'll learn:

* Multiple ways to interact with an [Apptainer](https://apptainer.org/docs/user/latest/) container from the command line
* How to execute parallel jobs in Slurm using [job arrays](https://slurm.schedmd.com/sbatch.html#OPT_array)
* How to automate a hyperparameter sweep with [Weights & Biases (W&B)](https://wandb.ai/site) and Slurm arrays
* How to track these runs using the W&B dashboard and Slurm commands


## Today's Tutorial Agenda

* Authenticating Weights & Biases on NERSC [[link](running-wandb-on-hpc.md#step-1-authenticating-your-weights--biases-account-from-sherlock-or-farmshare)]
* Launching your W&B sweep configuration [[link](running-wandb-on-hpc.md#step-2-define-the-sweep-configuration)]
* Writing an `sbatch` script to run your sweep [[link](running-wandb-on-hpc.md#step-3-write-an-sbatch-script-to-run-your-sweep)]
* Using `array` to launch parallel W&B sweep agents [[link](running-wandb-on-hpc.md#step-4-use-array-in-sbatch-to-launch-perfectly-parallel-wandb-agents)]
* Monitoring your sweep on the W&B dashboard [[link](running-wandb-on-hpc.md#step-5-monitor-your-sweep-on-the-wb-dashboard)]
* Retrieving the best sweep configuration using `wandb` [[link](running-wandb-on-hpc.md#step-6-retrieve-the-best-configuration-directly-on-sherlock)]

## Want to Learn More?

- Are you just getting started with running batch jobs in HPC environments? [`sbatch` Tips and Tricks](https://github.com/MATRICS-Bootcamp/estimating-and-requesting-resources/blob/main/sbatch-tips-and-tricks.md)
- Are you interested in building an Apptainer container for your own software stack? [Build Your First Apptainer Container](https://github.com/stanford-sdss/package-management)
- Do you want to learn how to scale across multiple GPUs in Python? [Scaling up GPU Compute with PyTorch DDP](https://github.com/MATRICS-Bootcamp/scaling-up-cpus-and-gpus/blob/main/scaling_up_gpu_compute.ipynb)
