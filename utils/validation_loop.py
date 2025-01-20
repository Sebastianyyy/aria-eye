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
from .get_loss_fn import get_loss_fn

from utils.visualize import visualize_soft, visualize_hard, generate_heatmaps

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
    writer = SummaryWriter(
        log_dir=os.path.join(log_dir, "tensorboard_validation")
    ) # Create a subdirectory for tensorboard validation logs

    ###### Upload Model ######
    
    # Load the model dynamically based on the model name
    module_name = f"models.{config['model_name']}"
    model_module = importlib.import_module(module_name)
    model = model_module.get_model(config).to(device)

    # Upload weights
    weights_path = latest_weights_file_path(config)
    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError(f"No weights file found at {weights_path}")

    print(f"Loading model weights from {weights_path}")
    logging.info(f"Loading model weights from {weights_path}")
    state = torch.load(weights_path)
    model.load_state_dict(state["model_state_dict"])


    # Prepare the dataset for validation
    validation_set = AriaDataset(config, train=False)

    # DataLoader to load batches of data
    validation_loader = DataLoader(
        validation_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False
    )

    # Initialize loss functions dynamically
    loss_fn = get_loss_fn(config["loss_fn"])(config)
    metric_fns = {
        "mse": get_loss_fn("mse")(config),
        "weighted_mse": get_loss_fn("weighted_mse")(config),
        "rmse": get_loss_fn("rmse")(config),
        "mae": get_loss_fn("mae")(config),
        "f1": get_loss_fn("f1")(config),
        "precision": get_loss_fn("precision")(config),
        "recall": get_loss_fn("recall")(config),
        "accuracy": get_loss_fn("accuracy")(config),
    }
    metric_fns = metric_fns - {config["loss_fn"]} # Remove the loss function from the metrics

    total_loss = 0.0
    metrics = {key: 0.0 for key in metric_fns.keys()}
    num_batches = len(validation_loader)

    model.eval()
    with torch.no_grad():
        batch_iterator = tqdm(validation_loader, desc="Validating")
        for step, (X, y) in enumerate(batch_iterator):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)

            # Compute loss
            loss = loss_fn(y_hat, y)
            total_loss += loss.item()

            # Compute all metrics for reporting except the loss function
            for metric_name, metric_fn in metric_fns.items():
                metrics[metric_name] += metric_fn(y_hat, y).item()

            if config["task"] == "classification":
                y_hat = y_hat.flatten(2)
                y_hat_class = torch.argmax(y_hat, dim=-1)
                gt_x = torch.div(y_hat_class, config["shape"], rounding_mode="floor")
                gt_y = y_hat_class % config["shape"]
                y_hat = torch.stack([gt_x, gt_y], dim=-1) / config["shape"]

            if config["clip"]:
                y_hat = torch.clip(y_hat, 0, 1)

            # Perform visualizations if enabled
            if config["visualize_soft"]:
                heatmap = generate_heatmaps(y, config['frame_grabber'])
    
                for X_i, y_i, heatmap_i in zip(X, y, heatmap):
                    visualize_soft(
                        X=X_i.cpu(),
                        y=y_i.cpu(),
                        heatmaps=heatmap_i.cpu(),
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

            selected_loss = config["loss_fn"]
            postfix = {
                **{"loss": f"{loss.item():6.3f}"},
                **{metric: f"{value:6.3f}" for metric, value in metrics.items()}
            }
            batch_iterator.set_postfix(postfix)

    # Calculate average loss and metrics
    avg_loss = total_loss / num_batches
    avg_metrics = {metric: value / num_batches for metric, value in metrics.items()}


    print(f"Validation completed. Average Loss: {avg_loss:6.3f}")
    logging.info(f"Validation completed. Average Loss: {avg_loss:6.3f}")
    writer.add_scalar("Loss/validation", avg_loss)

    for metric, value in avg_metrics.items():
        print(f"Validation completed. Average {metric}: {value:6.3f}")
        logging.info(f"Validation {metric}: {value:6.3f}")
        writer.add_scalar(f"Metrics/{metric}", value)
    
    logging.info("Validation finished.") # Close TensorBoard writer
    writer.close()
