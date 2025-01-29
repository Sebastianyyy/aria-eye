import importlib
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from .AriaDataset import AriaDataset
from .config import (get_model_folder_path, get_transformations,
                     get_weights_file_path, latest_weights_file_path)
from .get_loss_fn import get_loss_fn, GridMetricLoss

from utils.visualize import visualize_soft, visualize_hard

def validation_loop(config):
    device = torch.device(config["device"])

    ###### SET SEED TO REPRODUCIBILITY ######
    seed = config["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    ###### CREATE LOGGING ######
    log_dir = get_model_folder_path(config)  # Path for storing logs
    log_filename = f"{config['model_name_log']}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=log_filepath,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    logging.info("Starting validation")

    ##### Initialize TensorBoard Writer ######
    tensorboard_log_dir = os.path.join(log_dir, "tensorboard_validation")  # Create a subdirectory for tensorboard validation logs
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # Print the TensorBoard link
    tensorboard_url = f"http://localhost:6006/"
    print(f"TensorBoard logs are being saved to: {tensorboard_log_dir}")
    print(f"To view TensorBoard, run: tensorboard --logdir={tensorboard_log_dir}")

    ###### Upload Model ######
    
    # Load the model dynamically based on the model name
    module_name = f"models.{config['model_name']}"
    model_module = importlib.import_module(module_name)
    model = model_module.get_model(config).to(device)

    #upload weights
    weights_path = latest_weights_file_path(config)
    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError(f"No weights file found at {weights_path}")

    print(f"Loading model weights from {weights_path}")
    logging.info(f"Loading model weights from {weights_path}")
    state = torch.load(weights_path)
    model.load_state_dict(state["model_state_dict"])


    # Prepare the dataset for validation
    validation_set = AriaDataset(config, train=False)

    #DataLoader to load batches of data
    validation_loader = DataLoader(
        validation_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False
    )

    model.eval()

    mse_loss_fn = torch.nn.MSELoss()
    mae_loss_fn = torch.nn.L1Loss()
    grid_metrics_loss_fn = GridMetricLoss(config)

    # Initialize accumulators for each metric
    total_rmse = 0.0
    total_precision_loss = 0.0
    total_recall_loss = 0.0
    total_accuracy_loss = 0.0
    total_fone_loss = 0.0
    total_mae_loss = 0.0

    num_batches = len(validation_loader)

    with torch.no_grad():
        batch_iterator = tqdm(validation_loader, desc="Validating")
        for step, (X, y) in enumerate(batch_iterator):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)

            if config["task"] == "classification":
                y_hat = y_hat.flatten(2)
                y_hat_class = torch.argmax(y_hat, dim=-1)
                gt_x = torch.div(y_hat_class, config["shape"], rounding_mode="floor")
                gt_y = y_hat_class % config["shape"]
                y_hat = torch.stack([gt_x, gt_y], dim=-1) / config["shape"]

            if config["clip"]:
                y_hat = torch.clip(y_hat, 0, 1)

           
            # Calculate each metric
            rmse_loss = torch.sqrt(mse_loss_fn(y_hat, y))
            mae_loss = mae_loss_fn(y_hat,y)
            grid_metrics_loss = grid_metrics_loss_fn(y_hat,y)
            precision_loss = grid_metrics_loss["precision"]
            recall_loss = grid_metrics_loss["recall"]
            accuracy_loss = grid_metrics_loss["accuracy"]
            fone_loss = grid_metrics_loss["f1"]

            # Accumulate metrics
            total_rmse += rmse_loss.item()
            total_precision_loss += precision_loss
            total_recall_loss += recall_loss
            total_accuracy_loss += accuracy_loss
            total_fone_loss += fone_loss
            total_mae_loss += mae_loss

            # Perform visualizations if enabled
            if config["visualize_soft"]:
                for X_i, y_i, y_hat_i in zip(X, y, y_hat):
                    visualize_soft(
                        X=X_i.cpu(),
                        y=y_i.cpu(),
                        heatmaps=y_hat_i.cpu(),
                        num_frames=config['frame_grabber'],
                        writer=writer,
                        step=step
                    )

            if config["visualize_hard"]:
                for X_i,y_i,y_hat_i in zip(X,y,y_hat):
                  visualize_hard(
                      X_i.cpu(),
                      y_i.cpu(),
                      y_hat_i.cpu(),
                      num_frames=config['frame_grabber'],
                      writer=writer, 
                      step=step
                  )

    # Calculate averages for all metrics
    avg_rmse = total_rmse / num_batches
    avg_precision_loss = total_precision_loss / num_batches
    avg_recall_loss = total_recall_loss / num_batches
    avg_accuracy_loss = total_accuracy_loss / num_batches
    avg_fone_loss = total_fone_loss / num_batches
    avg_mae_loss = total_mae_loss / num_batches

    print(f"Validation completed. Average RMSE: {avg_rmse:.4f}")
    print(f"Validation completed. Average Precision Loss: {avg_precision_loss:.4f}")
    print(f"Validation completed. Average Recall Loss: {avg_recall_loss:.4f}")
    print(f"Validation completed. Average Accuracy Loss: {avg_accuracy_loss:.4f}")
    print(f"Validation completed. Average F1 Loss:{avg_fone_loss:.4f}")
    print(f"Validation completed. Average MAE Loss:{avg_mae_loss:.4f}")

    logging.info(f"Validation completed. Average RMSE: {avg_rmse:.4f}")
    logging.info(f"Validation completed. Average Precision Loss: {avg_precision_loss:.4f}")
    logging.info(f"Validation completed. Average Recall Loss: {avg_recall_loss:.4f}")
    logging.info(f"Validation completed. Average Accuracy Loss: {avg_accuracy_loss:.4f}")
    logging.info(f"Validation completed. Average F1 Loss:{avg_fone_loss:.4f}")
    logging.info(f"Validation completed. Average MAE Loss:{avg_mae_loss:.4f}")


    # Log metrics to TensorBoard
    writer.add_scalar("Metrics/validation_rmse", avg_rmse)
    writer.add_scalar("Metrics/validation_precision_loss", avg_precision_loss)
    writer.add_scalar("Metrics/validation_recall_loss", avg_recall_loss)
    writer.add_scalar("Metrics/validation_accuracy_loss", avg_accuracy_loss)
    writer.add_scalar("Metrics/validation_fone_loss", avg_fone_loss)
    writer.add_scalar("Metrics/validation_mae_loss", avg_mae_loss)



    writer.close()

    logging.info("Validation finished.") # Close TensorBoard writer