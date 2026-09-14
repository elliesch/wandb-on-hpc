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
        n_jobs=-1,  # use all CPUs on the node, important when the full node is reserved
    )

    # Evaluate with 5-fold cross-validation
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    mean_accuracy = float(np.mean(scores))

    # Log the metric W&B will optimize
    wandb.log({"val_accuracy": mean_accuracy})

if __name__ == "__main__":
    main()
