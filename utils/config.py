from pathlib import Path

import torch
from torchvision import transforms as T


def get_config():
    """
    Returns a dictionary containing the configuration for the experiment.

    Returns:
        dict: Configuration parameters including model settings, dataset details,
            optimizer, scheduler, and other hyperparameters.
    """
    return {
        "task": "classification",  # (regression, classification, autoregressive)
        "batch_size": 16,  # Number of samples per batch
        "num_epochs": 7,  # Total number of epochs for training
        "input_image_size": 224,  # Input image dimensions (square: width = height)
        "optimizer": "AdamW",  # Type of optimizer to use (e.g., Adam, SGD, AdamW)
        "lr": 1e-3,  # Learning rate for the optimizer
        "weight_decay": 1e-10,  # Regularization term to prevent overfitting
        "scheduler": "CosineAnnealingLR",  # Type of learning rate scheduler
        "scheduler_t_max": 600,  # Num of epochs over which to decay the learning rate for scheduler
        "scheduler_eta_min": 0.0001,  # Minimum learning rate value for the scheduler
        "loss_fn": "kl_loss",  # Loss function to use (e.g., mse, weighted_mse)
        "seed": 42,  # Random seed for reproducibility of experiments
        "num_workers": -1,  # Number of workers for DataLoader (-1 = use all available CPU cores)
        "train_data": "data/downloads_train",  # Path to the train dataset
        "test_data": "data/downloads_test",  # Path to the test dataset
        "sample": 10,  # Sampling over video
        "frame_grabber": 3,  # Number of consecutive frames to grab
        "model_name": "resnet",  # Name of the model architecture to use
        "model_name_log": "resnet3d_heatmap",  # Name of the model log file
        "model_basename": "model_",  # Base name for saving and loading model weight files
        "preload": "latest",  # Preload setting to load weights: "latest", "none", or specific point
        "dataset_path": "data/test_data",  # Path to the dataset directory
        "device": "cuda:0",  # Device to use for training and evaluation (cuda:0 or cpu)
        "preprocess_data_path": "data/preprocess_data",  # Path to data ready to use in training
        # RESNET PARAMS
        "model_depth": 50,  # Depth of resnet.py
        "num_of_classes": 10000,  # If not classification task set to 2
        # CLASSIFICATION
        "shape": 100,  # grid size
    }


def get_transformations(config):
    """
    Generates data transformations for training and testing datasets.

    Parameters:
    - config (dict): A dictionary containing configuration values,
        specifically the key 'input_image_size' which defines the target size
        for resizing images.

    Returns:
    - tuple: A tuple containing:
        1. train_transform: Transformations to apply to training images.
        2. test_transform: Transformations to apply to testing images.
    """
    train_transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Lambda(lambda x: torch.rot90(x, k=-1, dims=(1, 2))),
        ]
    )

    test_transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )

    return train_transform, test_transform


def get_model_folder_path(config):
    """
    Generate the folder path for saving model logs.

    Args:
        config (dict): Configuration dictionary containing 'datasource'
                    and 'model_name_log' keys.

    Returns:
        str: A string representing the path to the model folder.
    """
    # Create a folder path using logs and model_name from the config dictionary
    return f"logs/{config['model_name_log']}"


def get_weights_file_path(config, epoch: str):
    """
    Constructs the path to a specific weights file based on the given epoch.

    Args:
        config (dict): Configuration dictionary containing experiment parameters.
        epoch (str): Epoch identifier (e.g., "latest" or a specific epoch number).

    Returns:
        str: Full path to the weights file for the specified epoch.
    """
    # Path to the model folder based on the log model name
    model_folder = get_model_folder_path(config)
    # Filename for the weights file, using the base name and epoch identifier
    model_filename = f"{config['model_basename']}{epoch}.pt"
    # Combine folder and filename to create the full path
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config):
    """
    Finds the latest weights file in the weights folder.

    Args:
        config (dict): Configuration dictionary containing experiment parameters.

    Returns:
        str or None: Path to the most recent weights file. Returns None if no file is found.
    """
    # Path to the model folder based on the log model name
    model_folder = get_model_folder_path(config)
    # Glob pattern to match weight files in the folder
    model_filename = f"{config['model_basename']}*"
    # Find all matching files
    weights_files = list(Path(model_folder).glob(model_filename))
    # If no files are found, return None
    if len(weights_files) == 0:
        return None
    # Sort the files and return the latest one
    weights_files.sort()
    return str(weights_files[-1])
