import wandb

def log_losses(losses: dict, step_type: str = "train"):
    """
    Log losses to wandb.
    
    Args:
        losses (dict): Dictionary of losses to log.
        step_type (str): Type of step (train/val).
    """
    for name, value in losses.items():
        wandb.log({f'{step_type}/{name}': value})

def log_imgs(imgs: dict, step_type: str = 'train'):
    wandb_imgs = [wandb.Image(img, "RGB", name) for name, img in imgs.items()]
    wandb.log({f'{step_type}/img': wandb_imgs})
