from omegaconf import OmegaConf
from diflow_tts import DiFlowTTS
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config-path", required=True, help="path to configuration file.")
    parser.add_argument("--exp-root", type=str, default=None, help="Root directory for experiments")
    parser.add_argument("--project-name", type=str, default=None, help="Project name")
    parser.add_argument("--run-name", type=str, default="default_run", help="Run name for logging")
    parser.add_argument("--wandb-id", type=str, default=None, help="Wandb id")
    parser.add_argument("--logging-method", type=str, default="tensorboard", help="Logging method")
    parser.add_argument("--ckpt-path", type=str, default=None, help="path to checkpoint")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = OmegaConf.load(args.config_path)

    assert args.logging_method in ["tensorboard", "wandb"], "Logging method should be either tensorboard or wandb!"

    run_path = os.path.join(args.exp_root, args.project_name, args.run_name)
    if args.logging_method == "wandb":
        logger = WandbLogger(project=args.project_name,
                             name=args.run_name, 
                             save_dir=run_path,
                             id=args.wandb_id,
                             resume="allow")
    else:
        logger = TensorBoardLogger(name=args.run_name,
                                   save_dir=run_path)

    model = DiFlowTTS(config.model)
    model.setup_optimizer(config.optim)

    config.dataset.vocab_size = config.model.codec_model.vocab_size
    model.setup_dataset(config.dataset)
    train_loader, val_loader = model.get_dataloader()

    checkpoint_callback = ModelCheckpoint(
        monitor="loss/val",
        mode="min",
        save_top_k=1,
        filename="best-loss-val-{epoch:04d}",
        save_last=True
    )

    trainer = pl.Trainer(
        devices=config.train.num_devices,
        logger=logger,
        default_root_dir=os.path.join(args.exp_root, args.project_name, args.run_name),
        accelerator=config.train.device,
        max_epochs=config.train.epochs,
        strategy='ddp_find_unused_parameters_true' if config.train.num_devices > 1 else "auto",
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)

if __name__ == "__main__":
    main()