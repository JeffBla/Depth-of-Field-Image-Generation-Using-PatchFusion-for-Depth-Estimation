import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from hydra import initialize, compose
from hydra.utils import instantiate
from src.data.datamodule import DoFDataModule
from src.dof_model import DoFModel


@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    pl.seed_everything(cfg.seed)

    # Initialize DataModule
    datamodule = DoFDataModule(cfg.data)

    # Initialize model
    model = DoFModel(cfg.model)

    # Initialize WandB logger
    wandb_logger = WandbLogger(project=cfg.project_name, name=cfg.run_name)

    # Initialize ModelCheckpoint callback
    checkpoint_filename = '{epoch}-{' + cfg.monitor_metric + ':.2f}'
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.checkpoint_dir,
                                          filename=checkpoint_filename,
                                          save_top_k=3,
                                          monitor=cfg.monitor_metric)

    # Initialize Trainer
    trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                         gpus=cfg.gpus,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback],
                         **cfg.trainer)

    # Train the model
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
