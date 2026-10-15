
import matplotlib

# Use a non-interactive backend so plots can be saved without
# waiting for a graphical window to close.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_training_loss(loss_history, save_path=None):
    """Plot and optionally save the training loss."""

    if len(loss_history) == 0:
        raise ValueError("Loss history cannot be empty.")

    if any(loss < 0 for loss in loss_history):
        raise ValueError("Loss values cannot be negative.")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        range(1, len(loss_history) + 1),
        loss_history,
        marker="o"
    )

    ax.set_title("Neural Network Training Loss")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
