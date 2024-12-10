import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.datamodule import DofDataModule
from src.dof_model import VQVAEPix2PixSystem as DoFModel
from src.dof_model_enhance import VQVAEPix2PixSystemEnhanced as DoFModelEnhanced

@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    trainConf = cfg.train or {}

    pl.seed_everything(cfg.seed)

    # Initialize DataModule
    datamodule = DofDataModule(cfg.seed, cfg.data)

    # Initialize model
    if trainConf.isEnhanced:
        if trainConf.isCheckpoint:
            model = DoFModelEnhanced.load_from_checkpoint(trainConf.checkpoint, modelConf=cfg.model, strict=False)
        else:
            model = DoFModelEnhanced(cfg.model)
    else:
        model = DoFModel(cfg.model)

    # Convert config to dict for wandb logging
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Initialize WandB logger with config
    wandb_logger = WandbLogger(
        project=cfg.project_name,
        name=cfg.run_name,
        config=config_dict,  # Log all hyperparameters
    )

    # Initialize ModelCheckpoint callback
    checkpoint_filename = '{epoch}-{' + trainConf.monitor_metric + ':.2f}'
    checkpoint_callback = ModelCheckpoint(dirpath=trainConf.checkpoint_dir,
                                          filename=checkpoint_filename,
                                          save_top_k=trainConf.save_top_k,
                                          monitor=trainConf.monitor_metric)

    # Initialize Trainer
    trainer = pl.Trainer(max_epochs=trainConf.max_epochs,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback])

    # Train the model
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
