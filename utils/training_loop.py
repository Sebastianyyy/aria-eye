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


# Validation function to evaluate the model's performance on the test setSS
def validate(model, test_loader, loss_fn, device, epoch):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation during validation
        # Create a progress bar for the validation loop using tqdm
        batch_iterator = tqdm(
            test_loader, desc=f"Validating Epoch {epoch:02d}", leave=False
        )

        for X, y in batch_iterator:
            X = X.to(device)
            y = y.to(device)

            y_hat = model(X)

            loss = loss_fn(y_hat, y)
            total_loss += loss.item()

            # Update the progress bar with the current batch loss
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

    # Calculate the average loss across the validation set
    avg_loss = total_loss / len(test_loader)
    return avg_loss


# Main training loop function
def training_loop(config):
    ###### SET SEED TO REPRODUCIBILITY ######
    seed = config["seed"]  # Seed from config
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False  # Disables optimization to ensure reproducibility

    ###### CREATE LOGGING ######
    log_dir = get_model_folder_path(config)  # Path for storing logs
    log_filename = f"{config['model_name_log']}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # Check if the log directory exists
    if not os.path.exists(log_dir):
        # Ensure the model folder exists if it doesn't
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Set up logging configuration to write logs to a file
        logging.basicConfig(
            filename=log_filepath,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

        # Log model configuration parameters
        logging.info("MODEL PARAMETERS")
        for key, value in config.items():
            logging.info(f"{key}: {value}")

        # Log Transformations
        train_compose, test_compose = get_transformations(config)
        logging.info("Train Images Transformations")
        for i, transform in enumerate(train_compose.transforms):
            logging.info(f"Train transform {i + 1}: {transform}")
        logging.info("Test Images Transformations")
        for i, transform in enumerate(test_compose.transforms):
            logging.info(f"Test transform {i + 1}: {transform}")

        print(f"Logging setup complete. Logs will be stored at {log_filepath}.")
    else:
        # Only set up logging if it's not already set up
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                filename=log_filepath,
                level=logging.INFO,
                format="%(asctime)s - %(message)s",
            )
        print(
            f"Log directory already exists: {log_dir}. Skipping logging setup, "
            "but logs will continue."
        )

    ###### DEFINE MODEL PARAMETERS ######
    print(f"Model {config['model_name']}")

    device = config["device"]  # Select the device from config
    print("Using device: ", device)

    # If using CUDA (GPU), print additional device info
    if device == "cuda:0":
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(
            f"Device memory: "
            f"{torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB"
        )
    device = torch.device(device)  # Set device for model

    # Load the model dynamically based on the model name
    module_name = f"models.{config['model_name']}"
    model_module = importlib.import_module(module_name)
    model = model_module.get_model(config).to(device)

    # Prepare the dataset for training and testing
    train_set = AriaDataset(config, train=True)
    test_set = AriaDataset(config, train=False)

    # DataLoader to load batches of data
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    # Initialize optimizer and scheduler
    optimizer = getattr(optim, config["optimizer"])(
        params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    scheduler = getattr(optim.lr_scheduler, config["scheduler"])(
        optimizer,
        T_max=config["scheduler_t_max"],
        eta_min=config["scheduler_eta_min"],
    )

    # Initialize the loss function
    loss_fn = getattr(torch.nn, config["loss_fn"])()

    ###### PRELOAD MODEL IF NEEDED ######
    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    model_filename = (
        latest_weights_file_path(config)
        if preload == "latest"
        else get_weights_file_path(config, preload) if preload else None
    )

    if model_filename:
        print(f"Preloading model {model_filename}")
        logging.info(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")

    ##### Initialize TensorBoard Writer ######
    writer = SummaryWriter(
        log_dir=os.path.join(log_dir, "tensorboard")
    )  # Create a subdirectory for tensorboard logs

    ##### TRAIN MODEL ######
    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(
            train_loader, desc=f"Processing Epoch {epoch:02d}"
        )  # Progress bar for training

        total_loss = 0.0
        for step, (X, y) in enumerate(batch_iterator):
            X = X.to(device)
            y = y.to(device)

            y_hat = (torch.zeros_like(y) + 0.5).to(device)

            loss = loss_fn(y_hat, y)
            total_loss += loss.item()
            batch_iterator.set_postfix(
                {"loss": f"{loss.item():6.3f}"}
            )  # Update progress bar with current loss

        logging.info(f"target {y}")
        logging.info(f"predicted {y_hat}")
        scheduler.step()  # Adjust learning rate

        # Log the average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch} - Avg Train Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)  # Log to TensorBoard
        print(f"Epoch {epoch} - Avg Train Loss: {avg_loss:.4f}")

        # Validation after each epoch
        val_loss = validate(model, test_loader, loss_fn, device, epoch)
        logging.info(f"Epoch {epoch} - Avg Test Loss: {val_loss:.4f}")
        writer.add_scalar("Loss/val", val_loss, epoch)  # Log to TensorBoard
        print(f"Epoch {epoch} - Avg Test Loss: {val_loss:.4f}")

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:03d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )
        logging.info(f"Saving model {model_filename}")

    logging.info("Training finished.")
    writer.close()  # Close the TensorBoard writer
