import warnings

from utils.config import get_config_validation
from utils.validation_loop import validation_loop

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config_validation()
    validation_loop(config)
