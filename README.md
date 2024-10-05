# Depth of Field Image Generation Using PatchFusion for Depth Estimation

Depth estimation

[Target Paper](https://zhyever.github.io/patchfusion/)

## Prerequirements

### Theory

[self-attention](https://medium.com/@x02018991/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98-self-attention-fa6897080a0a)

[self-attention example code](https://github.com/PaulEm6/Attention-is-all-you-need/blob/main/Training%20Scripts/simple_gpt.py)

[pix2pix](https://github.com/phillipi/pix2pix)

[Transformer](https://arxiv.org/abs/1706.03762)

[Swin Transformer](https://arxiv.org/abs/2103.14030)

## Detail

### Version Control

- Github:

### Project Structure

- Pytorch & lightning
- Hydra for Hyperparameter controlling

### Data Management

- DVC in Google Drive

### Model Versioning

- Weights & Biases

### Experiment Tracking

- Weights & Biases

### Environment

- conda

<!-- Continuous Integration/Continuous Deployment (CI/CD):

Set up automated testing using tools like GitHub Actions or Jenkins.
Implement automated model evaluation on test sets. -->

## Collaboration

For the training process specifically in a multi-person situation:

### Data Preparation

Divide data preprocessing tasks among team members.
uses the same data splits for training, validation, and testing.

### Model Development

Assign different model components or experiments to different team members.
Use modular design to allow easy integration of different components.

### Hyperparameter Tuning

Coordinate hyperparameter search efforts to avoid duplication.
Share and discuss results regularly to inform future experiments.

### Resource Management

If sharing computational resources, implement a scheduling system for GPU usage.
Consider using cloud platforms for scalable computing if local resources are limited.

#### Environment

The server in the Resource department. **Please be very very careful.**

### Result Analysis

Collaboratively analyze results and discuss insights.
Use visualization tools to share and compare model performance across experiments.

## File structure

```plaintext!

dof_generation/
│
├── data/
│   ├── raw/                  # Raw data, tracked by DVC
│   └── processed/            # Processed data, tracked by DVC
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── datamodule.py     # Lightning DataModule
│   ├── models/
│   │   ├── __init__.py
│   │   ├── patchfusion.py    # PatchFusion model
│   │   ├── vae.py            # VAE model
│   │   └── gan.py            # GAN model
│   ├── utils/
│   │   ├── __init__.py
│   │   └── metrics.py        # Custom metrics
│   └── dof_model.py          # Main Lightning Module
│
├── configs/
│   ├── config.yaml           # Base configuration
│   ├── data/
│   │   └── default.yaml      # Data configuration
│   ├── model/
│   │   └── default.yaml      # Model configuration
│   └── train/
│       └── default.yaml      # Training configuration
│
├── scripts/
│   └── train.py              # Training script
│
├── notebooks/
│   └── data_exploration.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   └── test_model.py
│
├── .dvcignore
├── .gitignore
├── environment.yml           # Conda environment file
├── README.md
└── requirements.txt

```
