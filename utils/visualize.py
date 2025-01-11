import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

unnormalize = transforms.Normalize(
    mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],  # Assuming mean=0.5, std=0.5
    std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
)

def visualize_hard(X, y, y_pred, num_frames=1, writer=None, step=0):
    """
    Save visualization results
    Args:
        X: image
        y: ground truth coordinates
        y_pred: predicted coordinates
        num_frames: config['frame_grabber']
        writer: tensorboard writer
        step: global step for TensorBoard logging
    """

    # Create a figure for plotting
    fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))  # Adjust figsize based on the number of frames

    for i in range(num_frames):
        # if frame_grabber > 1 - plot all examples
        if X.dim() == 4:
            image_tensor = X[i]  # Shape: [channels, height, width]
            point = y[i] * 224  # Assuming y contains normalized coordinates in [0, 1]
            pred_point = y_pred[i] * 224  # Scale predicted coordinates
        else:
            image_tensor = X
            point = y[0] * 224  # Scale normalized coordinates
            pred_point = y_pred[0] * 224  # Scale predicted coordinates

        # Unnormalize the image if it was normalized during preprocessing
        image_tensor = unnormalize(image_tensor)

        # Convert the tensor to numpy and permute dimensions
        image_array = image_tensor.permute(1, 2, 0).numpy()  # Convert to [height, width, channels]
        image_array = image_array.clip(0, 1)

        # Plot the image on the respective subplot
        ax = axes[i] if num_frames > 1 else axes
        ax.imshow(image_array)
        ax.scatter(point[0].item(), point[1].item(), c="green", s=50, label="GT")

        ax.scatter(pred_point[0].item(), pred_point[1].item(), c="yellow", s=50, label="Pred")

        ax.legend()
        ax.axis("off")

    # Adjust layout
    plt.tight_layout()

    # Save the figure to TensorBoard
    if writer is not None:
        writer.add_figure("Visualization", fig, global_step=step)
    plt.show()

def visualize_soft(X, y, heatmaps, num_frames=1, writer=None, step=0):
    """
    Save visualization results with heatmaps overlaid on images.

    Args:
        X: Image tensor [batch_size, channels, height, width].
        y: Ground truth coordinates [batch_size, num_frames, 2].
        y_pred: Predicted coordinates [batch_size, num_frames, 2].
        heatmaps: Heatmaps tensor [batch_size, num_frames, height, width].
        num_frames: Number of frames to visualize.
        writer: TensorBoard writer.
        step: Global step for TensorBoard logging.
    """
    # Create a figure for plotting
    fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))  # Adjust figsize based on num_frames

    for i in range(num_frames):
        # Select the i-th frame
        if X.dim() == 4:  # Handle batch of frames
            image_tensor = X[i]  # Shape: [channels, height, width]
            point = y[i] * 224  # Ground truth (scaled to image size)
            heatmap = heatmaps[i, 0]  # Corresponding heatmap
        else:  # Handle single frame
            image_tensor = X
            point = y[0] * 224
            heatmap = heatmaps[0, 0]

        image_tensor = unnormalize(image_tensor)
        image_array = image_tensor.permute(1, 2, 0).numpy()  # Convert to [height, width, channels]
        image_array = image_array.clip(0, 1)

        # Overlay heatmap on the image
        ax = axes[i] if num_frames > 1 else axes  # Handle case where num_frames == 1
        ax.imshow(image_array, alpha=0.8)
        ax.imshow(heatmap.numpy(), cmap="hot", alpha=0.5)  # Heatmap overlay

        # Plot ground truth and predicted points
        ax.scatter(point[0].item(), point[1].item(), c="green", s=50, label="GT")

        ax.legend()
        ax.axis("off")

    # Adjust layout
    plt.tight_layout()

    # Save the figure to TensorBoard if a writer is provided
    if writer is not None:
        writer.add_figure("Visualization", fig, global_step=step)

    plt.show()