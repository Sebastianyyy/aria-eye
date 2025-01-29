import warnings

import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
from .AriaDataset import AriaDataset
from .config import (get_model_folder_path, get_transformations,
                     get_weights_file_path, latest_weights_file_path)

def generate_heatmap(gaze_points, heatmap_bins, log_dir, dataset_name, model_basename, preload):
    """
    Generate and save a heatmap of gaze points, with Gaussian smoothing.
    """
    # Define a violet-green colormap similar to DINO self-attention visualizations
    violet_green_cmap = "cividis"

    gaze_points = np.array(gaze_points)  # Convert to numpy array
    heatmap, xedges, yedges = np.histogram2d(
        gaze_points[:, 0], gaze_points[:, 1], bins=heatmap_bins, range=[[0, 224], [0, 224]]
    )

    # Normalize heatmap values to [0, 255] for GaussianBlur compatibility
    heatmap = heatmap / heatmap.max() * 255
    heatmap = heatmap.astype(np.uint8)

    # Apply Gaussian blur for smoothing
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), sigmaX=5, sigmaY=5)

    # Normalize heatmap back to original scale
    heatmap = heatmap / 255 * heatmap.max()

    # Plot the heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(
        heatmap.T, origin="lower", cmap=violet_green_cmap, interpolation="nearest",
        extent=[0, 224, 0, 224]
    )
    plt.title(f"Gaze Heat Map {preload}")
    plt.xlabel("X")
    plt.ylabel("Y")
    heatmap_path = os.path.join(log_dir, f"{dataset_name}_{model_basename}{preload}_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Gaze heatmap saved to {heatmap_path}")



def generate_video(config):
    """
    Generate a video for the entire dataset showing predictions vs ground truth.
    Video is saved in logs/model_name as [dataset_name]_[model_version]
    """
    device = torch.device(config["device"])

    # Set seed for reproducibility
    seed = config["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    log_dir = get_model_folder_path(config)
    os.makedirs(log_dir, exist_ok=True)
    
    dataset_name = os.path.basename(config["test_data"]).replace(" ", "_")
    model_basename = config["model_basename"]
    preload = config["preload"]
    video_name = f"{dataset_name}_{model_basename}{preload}.mp4"  # Combine dataset name and model version
    video_path = os.path.join(log_dir, video_name)

    module_name = f"models.{config['model_name']}"
    model_module = __import__(module_name, fromlist=[''])
    model = model_module.get_model(config).to(device)

    # Load model weights
    preload = config["preload"]
    weights_path = (
        latest_weights_file_path(config)
        if preload == "latest"
        else get_weights_file_path(config, preload) if preload else None
    )

    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError(f"No weights file found at {weights_path}")

    print(f"Loading model weights from {weights_path}")
    state = torch.load(weights_path)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Prepare dataset and DataLoader
    dataset = AriaDataset(config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    def unnormalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        return image_tensor * std + mean

    # OpenCV Video Writer
    video_format = config["video_format"]
    fourcc = cv2.VideoWriter_fourcc(*video_format)  
    video_writer = None

    # Initialize variables for heatmap
    generate_heatmap_flag = config.get("generate_heatmap", False)
    heatmap_bins = 224
    gaze_points = [] if generate_heatmap_flag else None
    gt_gaze_points = [] if generate_heatmap_flag else None

    # Loop through dataset
    FPS_amount = config["FPS"]
    ground_truth_color = config["ground_truth_color"] 
    predicted_color = config["predicted_color"] 

    with torch.no_grad():
        for idx, (X, y) in enumerate(tqdm(data_loader, desc="Generating Video")):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            if config["task"] == "classification":
                y_pred = y_pred.flatten(2)
                y_pred_class = torch.argmax(y_pred, dim=-1)
                gt_x = torch.div(y_pred_class, config["shape"], rounding_mode="floor")
                gt_y = y_pred_class % config["shape"]
                y_pred = torch.stack([gt_x, gt_y], dim=-1) / config["shape"]

            if config["clip"]:
                y_pred = torch.clip(y_pred, 0, 1)
            
            # Collect gaze points for heatmap
            if generate_heatmap_flag:
                    gt_gaze_points.extend(y[0,::4].cpu().numpy() * 224)
                    gaze_points.extend(y_pred[0,::4].cpu().numpy()* 224)

            # Process each temporal frame in the batch
            for t in range(X.size(1)):  # Iterate over the temporal dimension
                temporal_tensor = X[0, t].cpu()  # [channels, height, width]
                temporal_tensor = unnormalize(temporal_tensor)

                # Convert to numpy array for visualization
                image_array = temporal_tensor.permute(1, 2, 0).numpy().clip(0, 1)  # [height, width, channels]
                image_array = (image_array * 255).astype(np.uint8)  # Convert to uint8 for OpenCV

                # Determine ground truth and predictions for this frame
                if t < X.size(1) - 1:  # Earlier frames
                    point = None  # No ground truth for earlier frames
                    pred_point = None
                else:  # Final frame (prediction target)
                    point = y[0, t - 1] * 224
                    pred_point = y_pred[0, t - 1] * 224



                # Draw the ground truth and prediction on the frame
                frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
                if point is not None:
                    cv2.circle(frame, (int(point[0].item()), int(point[1].item())), 5, ground_truth_color , -1) 
                if pred_point is not None:
                    cv2.circle(frame, (int(pred_point[0].item()), int(pred_point[1].item())), 5, predicted_color, -1) 

                # Initialize the video writer
                if video_writer is None:
                    height, width, _ = frame.shape
                    video_writer = cv2.VideoWriter(video_path, fourcc, FPS_amount, (width, height)) 

                # Write the frame to the video
                video_writer.write(frame)
            if idx == 20: break 

    # Release the video writer
    if video_writer is not None:
        video_writer.release()

    print(f"Video for entire dataset saved to {video_path}")

    # Call heatmap generation function if enabled
    if generate_heatmap_flag:
        generate_heatmap(gaze_points, heatmap_bins, log_dir, dataset_name, model_basename, preload)
        generate_heatmap(gt_gaze_points, heatmap_bins, log_dir, dataset_name, model_basename, preload + "_gt")


