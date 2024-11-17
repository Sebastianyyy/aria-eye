import uuid
import os
import logging
from torch.utils.tensorboard import SummaryWriter


def prepare_output_and_logger(args):
    if not args.output_dir:
        unique_str = str(uuid.uuid4())
        args.output_dir = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(args.output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Optional: prints to console as well
        ]
    )
    logger = logging.getLogger()

    logger.info(f"Output directory created: {args.output_dir}")

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.output_dir)
        logger.info("TensorBoard initialized")
    else:
        logger.warning("TensorBoard not available: not logging progress")

    return tb_writer, logger


def training_report(tb_writer, logger, iteration, losses_dict, elapsed, train=True):
    name_exp = "train" if train else "test"
    log_message = f"Iteration {iteration} ({name_exp}): "
    if tb_writer:
        for k, v in losses_dict.items():
            tb_writer.add_scalar(f"{name_exp}_loss/{k}", v.item(), iteration)
            log_message += f"{k}: {v.item():.4f} "
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    log_message += f"Elapsed time: {elapsed:.2f}s"
    logger.info(log_message)
