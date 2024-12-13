import cv2
import hydra
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.data.datamodule import DofDataModule
from src.dof_model import VQVAEPix2PixSystem as DoFModel
from src.dof_model_enhance import VQVAEPix2PixSystemEnhanced as DoFModelEnhanced

@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    predictConf = cfg.predict or {}

    pl.seed_everything(cfg.seed)

    # Initialize DataModule
    datamodule = DofDataModule(cfg.seed, predictConf.data)

    # Initialize model
    if predictConf.isEnhanced:
        model = DoFModelEnhanced.load_from_checkpoint(predictConf.checkpoint, modelConf=cfg.model, strict=False)
    else:
        model = DoFModel.load_from_checkpoint(predictConf.checkpoint, modelConf=cfg.model, strict=False)

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

    # Predict using the model
    predictions = trainer.predict(model, datamodule=datamodule)

    # Save predictions
    # Create output directory
    save_dir = Path(predictConf.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert tensor to OpenCV format and save
    idx = 0
    for pred in predictions:
        if isinstance(pred, torch.Tensor):
            # Ensure the tensor is on CPU and convert to numpy
            pred = pred.cpu().squeeze().numpy()  # Remove batch dimension if present
            
            # If tensor is in range [-1, 1], convert to [0, 255]
            if pred.min() < 0:
                pred = ((pred + 1) * 127.5).astype(np.uint8)
            else:
                pred = (pred * 255).astype(np.uint8)
        if len(pred.shape) == 3:
            if pred.shape[0] == 3:  # If channels are first
                    pred = np.transpose(pred, (1, 2, 0))  # CHW -> HWC
                    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
                    
                    # Save image
                    cv2.imwrite(str(save_dir / f'{idx:04}.jpg'), pred)
        else:    
            for generated_img in pred:
                    # If the image is in RGB format, convert to BGR for OpenCV
                    if generated_img.shape[0] == 3:  # If channels are first
                        generated_img = np.transpose(generated_img, (1, 2, 0))  # CHW -> HWC
                        generated_img = cv2.cvtColor(generated_img, cv2.COLOR_RGB2BGR)
                    
                    # Save image
                    cv2.imwrite(str(save_dir / f'{idx:04}.jpg'), generated_img)

                    idx += 1
                    
                    # # For wandb logging, convert back to RGB
                    # wandb_img = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
                    # wandb_logger.log({f'prediction_{idx}': wandb.Image(wandb_img)})
    
if __name__ == "__main__":
    main()