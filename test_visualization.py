
from visualization import plot_training_loss


loss_history = [0.95, 0.72, 0.48, 0.31, 0.18, 0.09]

plot_training_loss(
    loss_history,
    save_path="training_loss.png"
)

print("Visualization test passed!")
print("Plot saved as training_loss.png")
