import os
import torch
import pytorch_lightning
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import hydra

from src.dataset import RiverSegmentationDataset
from src.model import RiverModel
from src.utils import hp_from_cfg

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg):
    if cfg.train.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    
    seed_everything(cfg.train.seed)
    
    loggers = list()
    callbacks = list()
    if cfg.wandb.log:
        from pytorch_lightning.loggers import WandbLogger
        import wandb
        hyperparameters = hp_from_cfg(cfg)
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
        wandb.config.update(hyperparameters)
        wandb_logger = WandbLogger()
        loggers.append(wandb_logger)

    last_checkpoint_callback = ModelCheckpoint(
        filename="last",
        save_last=True,
    )
    min_loss_checkpoint_callback = ModelCheckpoint(
        monitor="train/loss",
        filename="min-loss",
        save_top_k=1,
        mode="min",
    )
    min_val_loss_checkpoint_callback = ModelCheckpoint(
        monitor="validation/loss",
        filename="min-val-loss",
        save_top_k=1,
        mode="min",
    )

    callbacks.extend([
        last_checkpoint_callback,
        min_loss_checkpoint_callback,
        min_val_loss_checkpoint_callback
    ])
    
    train_dataset = RiverSegmentationDataset(cfg, "train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=os.cpu_count()
    )

    test_dataset = RiverSegmentationDataset(cfg, "test")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=os.cpu_count()
    )

    model = RiverModel(
        **cfg.model,
        lr=cfg.train.lr,
        log_images_every_train=cfg.wandb.log_images_every.train,
        log_images_every_validation=cfg.wandb.log_images_every.validation
    )

    print(model)

    trainer = pytorch_lightning.Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator="gpu",
        devices=cfg.train.gpus,
        log_every_n_steps=1,
        max_steps=cfg.train.steps,
        val_check_interval=cfg.test.interval,
        limit_val_batches=cfg.test.batches,
    )
    
    trainer.fit(model, train_dataloader, test_dataloader)

    if cfg.train.save_file != "":
        trainer.save_checkpoint(cfg.train.save_file)

if __name__ == "__main__":
    train()
