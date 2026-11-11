import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_jta_compact import batch_process_coords, collate_batch, create_dataset, get_datasets
from model_jta_pooled import create_model
from utils.metrics import MSE_LOSS
from utils.utils import AverageMeter, create_logger, load_config, load_default_config


def compute_loss(model, config, in_joints, out_joints, in_masks, out_masks, padding_mask,
                 epoch=None, mode='val', loss_last=True, optimizer=None):
    _, in_F, _, _ = in_joints.shape
    metamask = (mode == 'train')
    pred_joints = model(in_joints, padding_mask, metamask=metamask)
    loss = MSE_LOSS(pred_joints[:, in_F:], out_joints, out_masks)
    return loss, pred_joints


def evaluate_loss(model, dataloader, config):
    loss_avg = AverageMeter()
    model.eval()
    with torch.no_grad():
        for joints, masks, padding_mask in dataloader:
            in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(
                joints, masks, padding_mask, config)
            padding_mask = padding_mask.to(config["DEVICE"])
            loss, _ = compute_loss(model, config, in_joints, out_joints, in_masks, out_masks, padding_mask)
            loss_avg.update(loss.item(), len(in_joints))
    return loss_avg.avg


def adjust_learning_rate(optimizer, epoch, config):
    lr = config['TRAIN']['lr'] * (config['TRAIN']['lr_decay'] ** epoch)
    if 'lr_drop' in config['TRAIN'] and config['TRAIN']['lr_drop']:
        lr = lr * (0.1 ** (epoch // (config['TRAIN']['epochs'] * 4. / 5.)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    print('lr: ', lr)


def save_checkpoint(model, optimizer, epoch, config, filename, logger, min_val_loss=1e4):
    path = os.path.join(config['OUTPUT']['ckpt_dir'], filename)
    logger.info(f'Saving checkpoint to {path}.')
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'config': config,
        'min_val_loss': float(min_val_loss),
    }
    tmp = path + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)  # a power cut mid-save can no longer corrupt the previous checkpoint


def dataloader_for(dataset, config, **kwargs):
    return DataLoader(dataset,
                      batch_size=config['TRAIN']['batch_size'],
                      num_workers=config['TRAIN']['num_workers'],
                      collate_fn=collate_batch,
                      **kwargs)


def train(config, logger, experiment_name="", resume_ckpt=None):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']

    dataset_train = ConcatDataset(get_datasets(config['DATA']['train_datasets'], config, logger))
    dataloader_train = dataloader_for(dataset_train, config, shuffle=True)
    logger.info(f"Training on a total of {len(dataset_train)} annotations.")

    dataset_val = create_dataset(config['DATA']['train_datasets'][0], logger, split="val",
                                 track_size=(in_F + out_F), track_cutoff=in_F)
    dataloader_val = dataloader_for(dataset_val, config, shuffle=False)

    writer_name = experiment_name + "_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    writer_train = SummaryWriter(os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_TRAIN"))
    writer_valid = SummaryWriter(os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_VALID"))

    model = create_model(config, logger)

    if config["MODEL"]["checkpoint"] != "":
        logger.info(f"Loading checkpoint from {config['MODEL']['checkpoint']}")
        checkpoint = torch.load(os.path.join(config['OUTPUT']['ckpt_dir'], config["MODEL"]["checkpoint"]),
                                weights_only=False)
        model.load_state_dict(checkpoint["model"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['TRAIN']['lr'])

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_parameters} parameters.")

    start_epoch = 0
    min_val_loss = 1e4
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt['model'])
        optimizer.load_state_dict(resume_ckpt['optimizer'])
        start_epoch = resume_ckpt['epoch'] + 1
        min_val_loss = resume_ckpt.get('min_val_loss', 1e4)
        logger.info(f"Resuming from epoch {start_epoch} (best val ADE so far: {min_val_loss:.4f})")
        # a different shuffling seed per resumed epoch range, so it does not replay epoch 0's batch order
        torch.manual_seed(config['SEED'] + start_epoch)

    if start_epoch >= config["TRAIN"]["epochs"]:
        logger.info("Nothing left to train: the checkpoint already reached the configured number of epochs.")
        return

    log_every = 200
    for epoch in range(start_epoch, config["TRAIN"]["epochs"]):
        start_time = time.time()
        dataiter = iter(dataloader_train)
        loss_avg = AverageMeter()

        if config["TRAIN"]["optimizer"] == "adam":
            adjust_learning_rate(optimizer, epoch, config)

        train_steps = len(dataloader_train)

        for i in range(train_steps):
            model.train()
            optimizer.zero_grad()

            try:
                joints, masks, padding_mask = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader_train)
                joints, masks, padding_mask = next(dataiter)

            in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(
                joints, masks, padding_mask, config, training=True)
            padding_mask = padding_mask.to(config["DEVICE"])

            loss, pred_joints = compute_loss(model, config, in_joints, out_joints, in_masks, out_masks,
                                             padding_mask, epoch=epoch, mode='train', optimizer=None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["TRAIN"]["max_grad_norm"])
            optimizer.step()

            loss_avg.update(loss.item(), len(joints))

            if (i + 1) % log_every == 0 or i + 1 == train_steps:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (train_steps - i - 1)
                print(f"epoch {epoch} ({i + 1}/{train_steps}) | loss {loss_avg.avg:.4f} | "
                      f"elapsed {elapsed / 60:.1f} min | eta {eta / 60:.1f} min", flush=True)

            if config['dry_run']:
                break

        writer_train.add_scalar("loss", loss_avg.avg, epoch)

        val_loss = evaluate_loss(model, dataloader_val, config)
        writer_valid.add_scalar("loss", val_loss, epoch)

        val_ade = val_loss / 100
        print(f"epoch {epoch}: train loss {loss_avg.avg:.4f} | val loss {val_loss:.4f} | val ADE {val_ade:.4f}",
              flush=True)

        if val_ade < min_val_loss:
            min_val_loss = val_ade
            print('------------------------------BEST MODEL UPDATED------------------------------')
            print('Best val ADE: ', val_ade)
            save_checkpoint(model, optimizer, epoch, config, 'best_val_checkpoint.pth.tar', logger, min_val_loss)

        if config['dry_run']:
            break

        save_checkpoint(model, optimizer, epoch, config, 'last_checkpoint.pth.tar', logger, min_val_loss)
        print(f'time for epoch {epoch}: {(time.time() - start_time) / 60:.1f} min', flush=True)

    if not config['dry_run']:
        save_checkpoint(model, optimizer, epoch, config, 'checkpoint.pth.tar', logger, min_val_loss)
    logger.info("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name. Otherwise will use timestamp")
    parser.add_argument("--cfg", type=str, default="", help="Config name. Otherwise will use default config")
    parser.add_argument("--resume", type=str, default="", help="Path to a last_checkpoint.pth.tar to continue from")
    parser.add_argument('--dry-run', action='store_true', help="Run just one iteration")
    args = parser.parse_args()

    # read the resume checkpoint into memory first, before the experiment folder is touched
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)

    if args.cfg != "":
        cfg = load_config(args.cfg, exp_name=args.exp_name)
    else:
        cfg = load_default_config()

    cfg['dry_run'] = args.dry_run

    random.seed(cfg['SEED'])
    torch.manual_seed(cfg['SEED'])
    np.random.seed(cfg['SEED'])

    if torch.cuda.is_available():
        cfg["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
    else:
        cfg["DEVICE"] = "cpu"

    logger = create_logger(cfg["OUTPUT"]["log_dir"])
    logger.info("Initializing with config:")
    logger.info(cfg)

    train(cfg, logger, experiment_name=args.exp_name, resume_ckpt=resume_ckpt)