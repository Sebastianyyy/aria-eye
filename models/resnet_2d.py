import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet50Regressor(nn.Module):
    def __init__(self, output_dim=6):
        """
        ResNet18-based model for regression tasks.

        Args:
            pretrained (bool): Whether to initialize the model with pretrained weights.
            output_dim (int): Number of output dimensions (default: 6 for 3 (x, y) pairs).
        """
        super(ResNet50Regressor, self).__init__()
        # Load ResNet50
        self.base_model = resnet18(pretrained=False)
        # Remove the final classification layer (fc)
        in_features = self.base_model.fc.in_features
        # Replace the classification head with a regression head
        self.base_model.fc = nn.Linear(in_features, output_dim)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, 6), where each (x, y)
            pair corresponds to a channel.
        """
        num_objects = x.size(1)  # This should be 3 based on your description
        outputs = []

        # Loop over the objects (channels)
        for i in range(num_objects):
            single_object = x[:, i, :, :, :]
            output = self.base_model(single_object)  # Shape: (batch_size, output_dim)
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1)


def get_model(config):
    return ResNet50Regressor(config["num_of_classes"])
