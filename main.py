import argparse
import os
from pathlib import Path
import numpy as np
import time
import torch
import json
import datetime
import logging
from utils import loggers_utils, train_one_epoch, evaluate
from data import AriaSet

def get_args_parser():
    parser = argparse.ArgumentParser(
        'ARIA eye gaze estimation model', add_help=False)
    # Training parameters
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=1000, type=int)

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224,
                        type=int, help='images input size')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.025,
                        help='weight decay')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')

    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    # Dataset parameters
    parser.add_argument('--data-path', default='/images/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='ARIA', choices=['ARIA'],
                        type=str, help='Aria dataset')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for creating new dir, if not then it will be used \
                        for resuming')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--num_workers', default=10, type=int)
    return parser


def main(args):
    print(args)
    tb_writer, logger = loggers_utils.prepare_output_and_logger(args)
    # Set up the device
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train = AriaSet(args.data_path, is_train=True)
    dataset_val = AriaSet(args.data_path, is_train=False)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False
    )

    print(f"Creating model: {args.model}")
    
    ##########################################
    # MODEL CREATION
    model = create_model(
        args.model,

    )

    if args.finetune:
        print(f"Finetuning from {args.finetune}")

    model.to(device)

    ###########################################
    # OPTIMIZER AND SCHEDULER CREATION
    optimizer = create_optimizer(args, model)

    lr_scheduler, _ = create_scheduler(args, optimizer)

    ##########################################
    # LOSS FUNCTION
    criterion = create_loss(args)

    criterion = torch.nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    if args.resume:
        print(f"Resuming from {args.resume}")

    start_time = time.time()
    max_accuracy = 0.0
    # TRAINING LOOP
    for epoch in range(0, args.epochs):
        model.train()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch
        )

        lr_scheduler.step(epoch)



        if epoch % 10 == 9:
            test_stats = evaluate(data_loader_val, model, device)
            #LOGER OUTPUT
            
    if args.output_dir:
        checkpoint_path = Path(output_dir / 'checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': criterion,
        }, checkpoint_path)

    total_time = time.time() - start_time

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Aria eye gaze estimation', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


