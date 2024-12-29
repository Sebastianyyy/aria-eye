import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class OpticalFlowComputation(nn.Module):
    def __init__(self):
        super(OpticalFlowComputation, self).__init__()

    def forward(self, x_t, x_t_minus_1):
        if x_t_minus_1 is None:
            # Handle x0 case: zero flow or duplicate x_t
            batch_size, _, height, width = x_t.shape
            return torch.zeros(batch_size, 2, height, width, device=x_t.device)

        # Convert tensors to NumPy arrays
        x_t_np = x_t.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
        x_t_minus_1_np = x_t_minus_1.permute(0, 2, 3, 1).cpu().numpy()

        batch_size, height, width, _ = x_t_np.shape
        flow_batch = []

        for i in range(batch_size):
            # Convert images to grayscale for optical flow calculation
            gray_t = cv2.cvtColor(
                (x_t_np[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
            )
            gray_t_minus_1 = cv2.cvtColor(
                (x_t_minus_1_np[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
            )

            # Calculate dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                gray_t_minus_1, gray_t, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )  # Returns flow as (H, W, 2)
            flow_batch.append(flow)

        # Convert flow back to tensor
        flow_batch = np.stack(flow_batch, axis=0)  # (B, H, W, 2)
        flow_t = (
            torch.from_numpy(flow_batch).permute(0, 3, 1, 2).to(x_t.device)
        )  # B, 2, H, W)

        return flow_t


class ResNetWithOpticalFlow(nn.Module):
    def __init__(self, num_classes):
        super(ResNetWithOpticalFlow, self).__init__()

        # Load pretrained ResNet
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(
            5, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # Optical flow computation module
        self.optical_flow = OpticalFlowComputation()

    def forward(self, x):
        outputs = []

        for i in range(x.shape[1]):
            xt = x[:, i, :, :, :]
            xt_minus_1 = x[:, i - 1, :, :, :] if i != 0 else None

            # Compute optical flow
            flow = self.optical_flow(xt, xt_minus_1)

            # Concatenate original frame with flow
            x_concat = torch.cat(
                [xt, flow], dim=1
            )  # Shape: (batch_size, 5, height, width)

            # Forward through ResNet
            out = self.resnet(x_concat)
            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1)


def get_model(config):
    return ResNetWithOpticalFlow(config["num_of_classes"])
