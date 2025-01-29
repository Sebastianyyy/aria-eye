import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_fn(loss_fn):
    if loss_fn == "mse":
        return nn.MSELoss
    if loss_fn == "weighted_mse":
        return EdgeWeightedMSELoss
    if loss_fn == "kl_loss":
        return KLDivergenceLoss
    if loss_fn == "cross_entropy_map":
        return CrossEntropyMap


class EdgeWeightedMSELoss(nn.Module):
    def __init__(self, center=(0.5, 0.5)):
        """
        Edge-weighted MSE Loss: Heavily penalizes errors near the edges of the image.

        Args:
            center (tuple): Center point of the image (default: (0.5, 0.5)).
            weight_exponent (int): Exponent to apply to the weights (default: 2).
        """
        super().__init__()
        self.center = torch.tensor(center)

    def forward(self, y, y_hat):
        """
        Compute the edge-weighted MSE loss.

        Args:
            y (torch.Tensor): Ground truth tensor of shape (N, 2).
            y_hat (torch.Tensor): Predicted tensor of shape (N, 2).

        Returns:
            torch.Tensor: Edge-weighted MSE loss.
        """
        # Compute the element-wise squared error
        mse = (y - y_hat) ** 2

        # Calculate distance from the center for each ground truth point
        center = self.center.to(
            y.device
        )  # Ensure the center tensor is on the same device
        distances = torch.sqrt(
            torch.sum((y - center) ** 2, dim=1)
        )  # Distance to center

        # Use distances as weights (e.g., raised to the power of weight_exponent)
        weights = distances**2

        # Apply weights to the squared errors
        weighted_mse = mse.sum(dim=1) * weights

        # Normalize by the sum of weights
        loss = torch.sum(weighted_mse) / torch.sum(weights)
        return loss


def create_gt_heatmap(coords, grid_size=100, sigma=0.05):
    """
    Creates ground truth heatmaps with a Gaussian distribution centered at given coordinates.

    Args:
        coords (Tensor): Tensor of shape (batch_size, 3, 2) with (x, y) coordinates for each object.
        grid_size (int): Number of cells along each dimension (height, width of the heatmap).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: Heatmap of shape (batch_size, 3, grid_size, grid_size).
    """
    device = coords.device

    # Create a grid
    x = torch.linspace(0, 1, grid_size, device=device)
    y = torch.linspace(0, 1, grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # Shape: (grid_size, grid_size, 2)

    # Expand dimensions for broadcasting
    grid = grid.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, grid_size, grid_size, 2)

    # Adjust coords' shape for broadcasting
    coords = coords.unsqueeze(2).unsqueeze(2)  # Shape: (batch_size, 3, 1, 1, 2)

    # Compute Gaussian heatmaps for each object
    heatmap = torch.exp(
        -torch.sum((grid - coords) ** 2, dim=-1) / (2 * sigma**2)
    )  # Shape: (batch_size, 3, grid_size, grid_size)

    # Normalize heatmaps for each object to sum to 1
    heatmap = heatmap / heatmap.sum(dim=(2, 3), keepdim=True)
    return heatmap


def create_gt_hard_heatmap(coords, grid_size=100):
    """
    Creates ground truth heatmaps with a Gaussian distribution centered at given coordinates.

    Args:
        coords (Tensor): Tensor of shape (batch_size, TIME, 2) with (x, y) coordinates for each object.
        grid_size (int): Number of cells along each dimension (height, width of the heatmap)..
    Returns:
        torch.Tensor: Heatmap of shape (batch_size, TIME, grid_size, grid_size).
    """
    device = coords.device
    batch_size=coords.shape[0]
    time=coords.shape[1]
    heatmap = torch.zeros(batch_size, time, grid_size, grid_size, device=device)

    # Use advanced indexing to set the value 1 at the (x, y) positions
    x_coords = torch.round(coords[:, :, 0]*grid_size).long()  # Shape: (batch_size, time)
    y_coords = torch.round(coords[:, :, 1]*grid_size).long()  # Shape: (batch_size, time)

    # Create an index tensor for batch and time dimensions
    batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, time)  # Shape: (batch_size, time)
    time_idx = torch.arange(time).unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, time)

    # Use the batch_idx, time_idx, x_coords, and y_coords to set the heatmap values
    heatmap[batch_idx, time_idx, x_coords, y_coords] = 1
    return heatmap


class CrossEntropyMap(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grid_size = config.get('shape', 100)

    def forward(self, y_hat, coords):
        """
        Compute Softmax Loss between predicted map and ground truth corodinates.
        Args:
            y_hat (torch.Tensor): Predicted heatmap of shape (batch_size, TIME, grid_size, grid_size).
            coords (torch.Tensor): Ground truth coordinates of shape (batch_size, TIME, 2).

        Returns:
            torch.Tensor: Loss value.
        """
        # Create ground truth heatmaps for all objects
        
        gt=create_gt_hard_heatmap(coords, grid_size=y_hat.shape[2])
        
        # Flatten the heatmaps
        predicted_log = y_hat.view(y_hat.size(0), -1)
        
        true_prob = gt.view(gt.size(0),-1)
        
        loss_per_object=torch.nn.functional.cross_entropy(predicted_log,true_prob)
        return loss_per_object
        

class KLDivergenceLoss(nn.Module):
    def __init__(self, config):
        """
        KL-Divergence Loss for heatmaps.

        Args:
            grid_size (int): Number of cells along each dimension (height, width of the heatmap).
            sigma (float): std of the Gaussian distribution for the ground truth heatmap.
        """
        super(KLDivergenceLoss, self).__init__()
        self.grid_size = config.get('shape', 100)
        self.sigma = config.get('sigma', 0.05)

    def forward(self, y_hat, coords):
        """
        Compute KL-Divergence loss between predicted and ground truth heatmaps.

        Args:
            y_hat (torch.Tensor): Predicted heatmap of shape (batch_size, 3, grid_size, grid_size).
            coords (torch.Tensor): Ground truth coordinates of shape (batch_size, 3, 2).

        Returns:
            torch.Tensor: Loss value.
        """
        # Create ground truth heatmaps for all objects
        true_heatmap = create_gt_heatmap(
            coords, grid_size=self.grid_size, sigma=self.sigma
        )

        # Flatten the heatmaps
        predicted_log = F.log_softmax(
            y_hat.view(y_hat.size(0), y_hat.size(1), -1), dim=-1
        )
        true_prob = true_heatmap.view(true_heatmap.size(0), true_heatmap.size(1), -1)

        # Compute KL-Divergence for each object and average over objects
        loss_per_object = F.kl_div(predicted_log, true_prob, reduction="none").sum(
            dim=-1
        )
        loss = loss_per_object.mean()  # Average over batch and objects
        return loss

from sklearn.metrics import classification_report, accuracy_score

class GridMetricLoss(nn.Module):
    def __init__(self, config):
        """
        F1 Score Loss for heatmaps with a discretized grid.

        Args:
            config (dict): Configuration dictionary with the following keys:
                - "grid_size" (int): Number of cells along each dimension (height, width of the heatmap).
        """
        super(GridMetricLoss, self).__init__()
        self.grid_size = config["grid_size"]  # Number of patches in each row/column
        self.num_classes = self.grid_size ** 2  # Total number of bins (e.g., 64 for 8x8 grid)

    def map_batch_to_bins(self, coords):
        """
        Map batch of coordinates to bin indices.

        Args:
            coords (torch.Tensor): Tensor of shape (N, 2), with values in the range [0, 1].

        Returns:
            torch.Tensor: Tensor of shape (N,) containing bin indices.
        """
        # Ensure coordinates are within [0, 1]
        coords = torch.clamp(coords, 0, 1)

        # Compute grid indices for x and y dimensions
        x_indices = torch.floor(coords[:, 0] * self.grid_size).long()
        y_indices = torch.floor(coords[:, 1] * self.grid_size).long()

        # Compute bin indices
        bin_indices = x_indices * self.grid_size + y_indices

        return bin_indices

    def forward(self, y, y_hat):
        """
        Compute Precision, Recall, F1 Score, and Accuracy.

        Args:
            y (torch.Tensor): Ground truth tensor of shape (N, 2), with values between [0, 1].
            y_hat (torch.Tensor): Predicted tensor of shape (N, 2), with values between [0, 1].

        Returns:
            dict: A dictionary containing precision, recall, F1 score, and accuracy.
        """
        # Map ground truth and predictions to bin indices
        y_bins = self.map_batch_to_bins(y).cpu().numpy()  # Shape: (N,)
        y_hat_bins = self.map_batch_to_bins(y_hat).cpu().numpy()  # Shape: (N,)

        y_bins = y_bins.flatten()
        y_hat_bins = y_hat_bins.flatten()

        # Calculate metrics using scikit-learn
        report = classification_report(y_bins, y_hat_bins, output_dict=True, zero_division=0)
        accuracy = accuracy_score(y_bins, y_hat_bins)

        # Extract precision, recall, and F1-score for the weighted average (multiclass scenario)
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1 = report["weighted avg"]["f1-score"]

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }
