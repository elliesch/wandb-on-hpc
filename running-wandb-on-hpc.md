# Running Hyperparameter Sweeps with Weights & Biases

Selecting the right hyperparameters is one of the most time-consuming parts of building a machine learning or deep learning model. [Weights & Biases](https://wandb.ai) (W&B, or `wandb` in Python) streamlines this process with a hyperparameter sweep tool, which coordinates hyperparameter search across multiple cluster jobs and logs every result to a centralized dashboard.

In this tutorial, we'll configure a W&B sweep to tune a random forest classifier, using Slurm array jobs to run multiple automated search agents in parallel.

To get started, please sign up for a [Weights & Biases account](https://app.wandb.ai/login?signup=true). The free tier is usually sufficient for academic-scale research projects.

---

## Before We Start: Job Arrays for Running Parallel Jobs

Often in computing settings, we find that we need to run the same computing task (such as a simulation codebase or training algorithm) repeatedly with differing inputs. This kind of workflow is called "perfectly parallel," in which your coded architecture or algorithm stays the same and the only thing that changes from run to run are variables external to your coded architecture. 

In scientific computing this type of workflow is common, for example when you are cross-validating ML/AI models or a testing a grid of input variables on a simulation codebase. You could manually create jobs to test each set of inputs separately and create unique `sbatch` files for each input configuration, but this is time consuming. Slurm's `--array` flag lets you submit many copies of the same job while varying the inputs with a single `sbatch` file instead.

Each of these job copies is called a task, and every task is given a unique `$SLURM_ARRAY_TASK_ID` as an environment variable that can be used as an iterator across all the array jobs, allowing you to iterate over your collection of variable inputs. Let's see how Slurm arrays work with a concise toy example using `run_arrays.sh` to print `Hello World` from an array of CPUs in parallel.

```bash
#!/bin/bash
#SBATCH --job-name=array_intro              # Name your job
#SBATCH --output=logs/array_%A_%a.out       # .out file to capture outputs
#SBATCH --error=logs/array_%A_%a.err        # .err files to capture error
#SBATCH --qos=debug                         # NERSC docs provide a helpful flowchart
#SBATCH --constraint=cpu                    # Perlmutter requires cpu or gpu spec
#SBATCH --account=your-project              # NERSC project associated with this job
#SBATCH --time=00:05:00                     # Run time in HH:MM:SS
#SBATCH --mail-type=BEGIN,END,FAIL          # Receive emails with job status
#SBATCH --mail-user=your-email@email.gov

# Create log directory
mkdir -p logs

# Hello World from each array task
echo "Hello World! From array task ${SLURM_ARRAY_TASK_ID} in job ${SLURM_JOB_ID} on $(hostname)"
sleep 10
```

Let's edit `run_arrays.sh` to use your project and email address. Now that we have our `sbatch` defined, let's launch an array with four tasks on the `debug` qos with the following command:
```bash
sbatch --array=1-4 run_arrays.sh
```

While our arrays are launching, we can check their run status with:
```bash
squeue --me
```

Once our runs have completed, we can check on their outputs using the associated `JobID` by running:
```bash
cat logs/array_jobID_1.out
cat logs/array_jobID_2.out
cat logs/array_jobID_3.out
cat logs/array_jobID_4.out
```

We can see all the different `$SLURM_ARRAY_TASK_ID`s even though each array task launched using the same `sbatch` script. We'll use Slurm job arrays below along with Weights & Biases to launch a hyperparameter sweep grid. Before we do, let's get started with W&B on our system.

---

## Step 1: Authenticating Your Weights & Biases Account from NERSC

In order to sync whatever system we're working from to your Weights & Biases account, we'll use an API key. Let's walk through creating a W&B key together in [the API key section](https://wandb.ai/settings#apikeys) of your account settings. 

> **Important Note:** Make sure to save your API key somewhere that you can access in the future. You'll only get to see it once, but that's what you'll use to log in from systems like NERSC or NSF ACCESS.

Now that you have an API key, we can use it to authenticate W&B on NERSC using our handy W&B container. You'll only need to do this once because W&B will save your API key to a `~/.netrc` file in your NERSC `$HOME` directory.

To use W&B, we'll be using the python package `wandb`, and we'll access this using an Apptainer container configured in Python. To use Apptainer on Perlmutter, run the following command to access the global Apptainer package.
```bash
export PATH=${PATH}:/cvmfs/oasis.opensciencegrid.org/mis/apptainer/1.3.3/x86_64/bin
```

We'll start by downloading the Apptainer container from this GitHub repo with the following command.
```bash
apptainer pull oras://ghcr.io/elliesch/wandb:latest
```

We'll start by activating a shell that operates inside our container.
```bash
apptainer shell wandb_latest.sif
```

You can confirm that you're working from inside the container, because instead of showing the login node at the beginning of the line:
```bash
username@perlmutter:loginXX:/your/path>
```

The command line shows the container environment:
```bash
(wandbenv)
```

Now that you are working from inside the container, run:

```bash
wandb login
```

When prompted, paste the API key that you created above and hit enter. Type `exit` to leave the `wandb` container shell when you're done.

> **Tip:** Your `~/.netrc` file is stored in your home directory so that it will be automatically accessible for any jobs.

---

## Step 2: Set up Your Training Script

Hyperparameter sweeps are useful on ML and AI models because they allow you to sweep through a potential grid of hyperparameters for your model and find the hyperparameters that return the best performance. In this example, we have a python script, `wandb_train.py`, that trains a random forest classifier across a hyperparameter grid. In order to use `wandb` to track our training runs, we'll initialize a `wandb` run, and set up a `run.config` that will automatically populate W&B sweep agents from a hyperparameter grid that we configure. The W&B agents automatically log the cross-validation accuracy for every grid step back to your W&B account, allowing you to track the best fit configuration easily, even across a very large sweep grid.

> **Tip:** The `n_jobs=-1` argument tells `scikit-learn` to use all available CPU cores on the node, which is important for making full use of your Slurm resource request.

```python
import wandb
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def main():
    # Initialize a W&B run. Config values are injected by the sweep agent.
    run = wandb.init()
    config = run.config

    # Load data
    X, y = load_digits(return_X_y=True)

    # Build model using sweep-provided hyperparameters
    clf = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        random_state=42,
        n_jobs=-1,  # use all CPUs on the node
    )

    # Evaluate with 5-fold cross-validation
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    mean_accuracy = float(np.mean(scores))

    # Log the metric W&B will optimize
    wandb.log({"val_accuracy": mean_accuracy})

if __name__ == "__main__":
    main()
```

**Let's take a closer look at two key functions in this script:**
- `wandb.init()` indicates that `wandb` will use the sweep agent to inject the individual hyperparameter configuration for each sweep number. Hardcoding values here would bypass the sweep entirely and the same configuration would run every time. We will configure the sweep grid in the next step in a file called `sweep.yaml`.
- `wandb.log({"val_accuracy": ...})` sends the validation accuracy of each run back to W&B. The key name used here, `val_accuracy`, needs to match the metric that you specify in the `metric.name` field of your `sweep.yaml` in the next step, so that W&B knows which value to optimize for.


---

## Step 2: Define the Sweep Configuration

When training our model, we want to define the hyperparameter search space and tell W&B how to explore it. We can do this with a `sweep.yaml` file. These files can be more concise like the example in this tutorial, or more complex. W&B has great documentation, including [a guide on creating defining sweep configurations](https://docs.wandb.ai/models/sweeps/define-sweep-configuration).

In our `sweep.yaml`, we specify three things: which training script to run, which search strategy to use, and which metric to optimize. Here we use Bayesian optimization (`method: bayes`), which learns from the results of previous runs to make informed decisions about which hyperparameter combinations to try next. This makes our search more efficient than a random or exhaustive grid search, and allows us to save time in this tutorial. 

The `run_cap` field sets the total number of configurations that will be tried across all agents combined, so rather than exhaustively testing every possible combination of our three parameters, W&B will intelligently select 10 configurations to evaluate. Here we limit to a small number so the tutorial will run in reasonable time, but you can try this later without a cap or with a higher cap.

```yaml
program: wandb_train.py

method: bayes          # Options: bayes, random, grid
metric:
  name: val_accuracy
  goal: maximize

parameters:
  n_estimators:
    values: [50, 100, 200]
  max_depth:
    values: [3, 10, 20]
  min_samples_split:
    values: [2, 6, 10]

run_cap: 10            # Stop after 10 total runs across all agents
```

In order to link our sweep to the W&B dashboard online, we'll need to register our sweep. To do this, run the following command. It's okay to run this from the login node. You should replace `my-project-name` with the intended name of your project.

```bash
apptainer exec wandb_latest.sif \
    wandb sweep sweep.yaml --project my-project-name
```

W&B will print a link to your sweep that looks like `username/my-project-name/sweeps/sweepID`. Copy the full path of the sweep ID to somewhere you can access it because you'll need it for the next step. As an example, one of my sweep IDs was `nzabw5am` while testing this tutorial.

---

## Step 3: Write an Sbatch Script to run your Sweep

We want to write a sbatch script that will allow us to run a single sweep agent across a Slurm job array, one agent per task. Each agent pulls a hyperparameter configuration from `sweep.yaml` then runs `wandb_train.py` with those parameters. Once that particular configuration completes in an array ID, `wandb` reports the result back to the W&B dashboard, and repeats with another configuration from `sweep.yaml` until either the `run_cap` is reached or the job's time limit expires. A few things to note as you fill this script in: set `--cpus-per-task` to match the `n_jobs=-1` argument in your training script, make sure your `SWEEP_ID` follows the format `username/my-project-name/sweepID` without the `/sweeps/` segment that you copied over in the last step. The `WANDB_DIR=/tmp` command prevents a folder naming conflict inside the container.

Let's take a closer look at `run_wandb.sh` before we prepare to submit using an array in the next step.

```bash
#!/bin/bash
#SBATCH --job-name=wandb_sweep              # Name your job
#SBATCH --output=logs/sweep_%A_%a.out       # .out file to capture outputs
#SBATCH --error=logs/sweep_%A_%a.err        # .err files to capture error
#SBATCH --qos=debug                         # NERSC docs provide a helpful flowchart for this
#SBATCH --constraint=cpu                    # Perlmutter requires you to specify cpu or gpu mode
#SBATCH --account=nguest                    # NERSC project associated with this job
#SBATCH --nodes=1                           # Book jobs on the same node
#SBATCH --ntasks-per-node=1                 # Select 1 task/node and wandb will spawn the tasks
#SBATCH --cpus-per-task=128                 # Match n_jobs in train.py
#SBATCH --time=00:15:00                     # Run time in HH:MM:SS
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
```

---

## Step 4: Use `array` in Sbatch to Launch Perfectly Parallel `wandb` Agents

Rather than submitting one W&B agent at a time, we can use Slurm's `array` feature to launch multiple agents simultaneously. Each agent independently pulls configurations from the same sweep queue, so they work in parallel without duplicating any runs. When an agent completes a run, it automatically launches the next run, and when all 10 runs (set by `run_cap`) are complete, W&B closes down the compute so you don't have to worry about managing the distribution. `wandb` handles that coordination automatically for us, so we can launch our sweep in just one line. In this case, we're launching 2 agents in compliance with Perlmutter's `debug` QoS, using an array of `1-2`.

```bash
sbatch --array=1-2 run_wandb.sh
```

The more agents you choose to run, the longer your job will queue, but in large jobs, you'll likely want more than two agents.

**Some approximate scaling recommendations (though this will depend on your cross validation grid):**

| Sweep size | Suggested agent count |
|---|---|
| < 20 runs | 2–4 |
| 20–60 runs | 4–10 |
| 60+ runs | 8–16 (watch your allocation) |

---

## Step 5: Monitor Your Sweep on the W&B Dashboard!

Open your W&B project dashboard at `https://wandb.ai/username/my-project-name`. As runs complete you can start to see how various parameters affect the validation accuracy. When running a Random Forest, these three panels can be useful for this kind of sweep:

- **Parallel coordinates plot**: visualizes how combinations of parameter values connect to `val_accuracy`
- **Parameter importance**: shows which of your hyperparameters has the most influence on `val_accuracy`
- **Optimizer Bar Chart**: shows the `val_accuracy` of each sweep

While your sweep is running, you can check your job status on Perlmutter with:

```bash
squeue --me
```

And look at any log in your `logs` dir for a specific job and array ID to see more granular sweep progress:

```bash
cat logs/sweep_<job-id>_<array-id>.out
```

---

## Step 6: Retrieve the Best Configuration Directly on NERSC

Once the sweep finishes, you can find the best run in the W&B Dashboard, or you can use the API from a login node to print out the best hyperparameter configutaion and the validation accuracy that configuration achieved.

Let's launch a shell from within our container to look more closely:
```bash
apptainer shell wandb_latest.sif
```

From here we'll launch Python:
```bash
python
```

And run the following commands to print out the best validation accuracy and hyperparameter configuration, filling in with your specific `username/my-project-name/sweepID`.
```python
import wandb
api = wandb.Api()
sweep = api.sweep("username/my-project-name/sweepID")
best_run = sweep.best_run()
print("Best val_accuracy:", best_run.summary["val_accuracy"])
print("Best config:", best_run.config)
```

---

## Using W&B in Your Workflow: Estimating Resources for Your Sweep on NERSC

On NERSC, the above submission charges your project allocation. Let's calculate our cost ([NERSC calculator](https://docs.nersc.gov/jobs/cost/)):
```
2 tasks X 128 cores      = 256 cores
256 / 128 cores per node = 2 node-equivalents
X
0.25 hours (15 min)      = 0.5 NERSC hours
```
You actual charge will be lower than this, since the job script exits once the W&B agents complete the `run_cap`. 

Above we booked 128 cpus per array task. We did this on purpose, in order to advantageously use all the physical cores on the `debug` nodes that we booked for each task. We can do this easily with `sklearn` because `njobs=-1` will automatically run our Random Forest model across all cores. If you are working on a QoS that allows for shared nodes, you might choose this differently, since jobs that ask for less resources can be schedule more flexibly. Typically, in a shared setting the more cpus that you request, the longer you will wait in queue. 

So how do we decide our job resources? 
Assuming that we are not on a shared queue, our focus becomes to take advantage of all the cores we have requested while minimizing our time on the machine, since larger requests will wait in queue longer for the same priority levels. To estimate your time, pick a representative sweep configuration, and manually test the time this configuration takes (For quick profiling the `time` command is very handy). Multiply the time that was required to complete your single run by the number you selected as your `run_cap`, and then add some extra time as a buffer, so your work isn't lost. As you become more comfortable with HPC, you will have a better sense of this buffer for your individual work, but when people are first starting out, a 25% buffer is a good starting point. 

> **Important Note:** W&B does not checkpoint or resume an individual grid configuration run if it is interrupted. For that reason, we aren't running on the `debug_preempt` queue, even though it is more cost-effective, and we make sure to overestimate our run time so we don't lose any runs. See [NERSC's W&B example](https://github.com/NERSC/nersc-dl-wandb) for a more detailed exploration of running W&B for GPU settings.

---

# The Distributed W&B Workflow Summarized in Three Steps:

1. Start by registering your W&B sweep:
   
```bash
apptainer exec wandb_latest.sif \
    wandb sweep sweep.yaml --project my-project-name
```

2. Launch parallel W&B sweep agents using Slurm job arrays:

```bash
sbatch --array=1-2 run_wandb.sh
```

3. Monitor results on the W&B Dashboard:

Go to [https://wandb.ai/username/my-project-name](https://wandb.ai/username/my-project-name) to monitor your sweep and configure your tiles.

