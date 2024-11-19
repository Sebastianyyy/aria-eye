from pathlib import Path
from torchvision import transforms as T

def get_config():
    """
    Returns a dictionary containing the configuration for the experiment.

    Returns:
        dict: Configuration parameters including model settings, dataset details, 
            optimizer, scheduler, and other hyperparameters.
    """
    return {
        "batch_size": 128,                          # Number of samples per batch
        "num_epochs": 100,                          # Total number of epochs for training
        "input_image_size": 32,                     # Input image dimensions (square: width = height)
        "optimizer": "AdamW",                       # Type of optimizer to use (e.g., Adam, SGD, AdamW)
        "lr": 10**-4,                               # Learning rate for the optimizer
        "weight_decay": 0,                          # Regularization term to prevent overfitting
        "scheduler": "CosineAnnealingLR",           # Type of learning rate scheduler
        "scheduler_t_max": 50,                      # Number of epochs over which to decay the learning rate for scheduler
        "scheduler_eta_min": 0.0001,                # Minimum learning rate value for the scheduler
        "loss_fn": "CrossEntropyLoss",              # Loss function to use (e.g., CrossEntropyLoss, MSELoss)
        "seed": 42,                                 # Random seed for reproducibility of experiments
        "num_workers": -1,                          # Number of workers for DataLoader (-1 = use all available CPU cores)
        "datasource": "test_dataset",               # Name of the dataset being used (placeholder until the dataset is ready)
        "model_name": "CNN",                        # Name of the model architecture to use
        "model_basename": "model_",                 # Base name for saving and loading model weight files
        "preload": "latest",                        # Preload setting to load weights (e.g., "latest", "none", or specific checkpoint)
        "dataset_path": "data/test_data",           # Path to the dataset directory
        "device": "cuda:0",                         # Device to use for training and evaluation ("cuda:<ID>" for GPU, "cpu" for CPU)
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
    train_transform = T.Compose([
        T.Resize(size=(config["input_image_size"], config["input_image_size"])),
        T.RandomRotation(degrees=45),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.05),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomGrayscale(p=0.33),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])

    test_transform = T.Compose([
        T.Resize(size=(config["input_image_size"], config["input_image_size"])),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])
    
    return train_transform, test_transform


def get_model_folder_path(config):
    return f"logs/{config['datasource']}_{config['model_name']}"


def get_weights_file_path(config, epoch: str):
    """
    Constructs the path to a specific weights file based on the given epoch.

    Args:
        config (dict): Configuration dictionary containing experiment parameters.
        epoch (str): Epoch identifier (e.g., "latest" or a specific epoch number).

    Returns:
        str: Full path to the weights file for the specified epoch.
    """
    # Path to the model folder based on the datasource and model name
    model_folder = get_model_folder_path(config)
    # Filename for the weights file, using the base name and epoch identifier
    model_filename = f"{config['model_basename']}{epoch}.pt"
    # Combine folder and filename to create the full path
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config):
    """
    Finds the latest weights file in the weights folder.

    Args:
        config (dict): Configuration dictionary containing experiment parameters.

    Returns:
        str or None: Path to the most recent weights file. Returns None if no file is found.
    """
    # Path to the model folder based on the datasource and model name
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
