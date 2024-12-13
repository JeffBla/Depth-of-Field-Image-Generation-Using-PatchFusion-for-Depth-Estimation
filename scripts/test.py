import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.data.datamodule import DofDataModule
from src.dof_model import VQVAEPix2PixSystem as DoFModel
from src.dof_model_enhance import VQVAEPix2PixSystemEnhanced as DoFModelEnhanced

@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    testConf = cfg.test or {}

    pl.seed_everything(cfg.seed)

    # Initialize DataModule
    datamodule = DofDataModule(cfg.seed, cfg.data)

    # Initialize model
    if testConf.isEnhanced:
        model = DoFModelEnhanced.load_from_checkpoint(testConf.checkpoint, modelConf=cfg.model, strict=False)
    else:
        model = DoFModel.load_from_checkpoint(testConf.checkpoint, modelConf=cfg.model, strict=False)

    # Convert config to dict for wandb logging
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Initialize WandB logger with config
    wandb_logger = WandbLogger(
        project=cfg.project_name,
        name=cfg.run_name,
        config=config_dict,  # Log all hyperparameters
    )

    # Initialize Trainer
    trainer = pl.Trainer(logger=wandb_logger)

    # Test the model
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    main()