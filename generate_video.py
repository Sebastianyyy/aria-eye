import warnings

from utils.config import get_config_validation
from utils.video_generator import generate_video

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config_validation()
    generate_video(config)
