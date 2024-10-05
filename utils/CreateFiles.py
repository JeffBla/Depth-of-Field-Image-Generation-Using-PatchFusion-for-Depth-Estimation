import os
import re

# # Define the tree structure as a multi-line string
# file_structure = """
# dof_generation/
# │
# ├── data/
# │   ├── raw/                  # Raw data, tracked by DVC
# │   └── processed/            # Processed data, tracked by DVC
# │
# ├── src/
# │   ├── data/
# │   │   ├── __init__.py
# │   │   └── datamodule.py     # Lightning DataModule
# │   ├── models/
# │   │   ├── __init__.py
# │   │   ├── patchfusion.py    # PatchFusion model
# │   │   ├── vae.py            # VAE model
# │   │   └── gan.py            # GAN model
# │   ├── utils/
# │   │   ├── __init__.py
# │   │   └── metrics.py        # Custom metrics
# │   └── dof_model.py          # Main Lightning Module
# │
# ├── configs/
# │   ├── config.yaml           # Base configuration
# │   ├── data/
# │   │   └── default.yaml      # Data configuration
# │   ├── model/
# │   │   └── default.yaml      # Model configuration
# │   └── train/
# │       └── default.yaml      # Training configuration
# │
# ├── scripts/
# │   └── train.py              # Training script
# │
# ├── notebooks/
# │   └── data_exploration.ipynb
# │
# ├── tests/
# │   ├── __init__.py
# │   ├── test_data.py
# │   └── test_model.py
# │
# ├── .dvcignore
# ├── .gitignore
# ├── environment.yml           # Conda environment file
# ├── README.md
# └── requirements.txt
# """


def parse_tree_structure(tree_str):
    """
    Parses a tree-like string structure and returns a list of tuples
    containing (depth, name, is_dir).
    """
    lines = tree_str.strip().split('\n')
    structure = []
    for line in lines:
        # Skip empty lines or lines with only tree symbols
        if not line.strip() or set(line.strip()) <= set('│'):
            continue

        # Remove leading tree symbols and extract indentation
        match = re.match(r'^((?:├── |└── |│ |\s)*)\s*(.*?)(\s+#.*)?$', line)
        if not match:
            continue
        indent, name, _ = match.groups()
        # Determine depth based on indentation
        depth = 0
        if indent is not None:
            depth = len(
                indent.replace('│ ', ' ').replace('├── ',
                                                  ' ').replace('└── ', ' '))
        # Determine if it's a directory
        is_dir = name.endswith('/')
        # Clean the name
        name = name.rstrip('/').strip()
        structure.append((depth, name, is_dir))
    return structure


def create_structure(structure, base_path='.'):
    """
    Creates directories and files based on the parsed structure.
    """
    path_stack = [base_path
                  ]  # Stack to keep track of current path at each depth
    for depth, name, is_dir in structure:
        # Ensure the path stack is the correct size
        if depth + 1 > len(path_stack):
            path_stack.append(name)
        else:
            path_stack = path_stack[:depth + 1]
            path_stack[-1] = name
        # Build the full path
        full_path = os.path.join(base_path, *path_stack[:depth + 1])
        if is_dir:
            os.makedirs(full_path, exist_ok=True)
            print(f"Directory created: {full_path}")
        else:
            # Create the file if it doesn't exist
            if not os.path.exists(full_path):
                with open(full_path, 'w') as f:
                    pass  # Create an empty file
                print(f"File created: {full_path}")


def main():
    with open('FileStructure.txt', 'r') as f:
        file_structure = f.read()
    # Parse the tree structure
    structure = parse_tree_structure(file_structure)

    # Define the base directory (current directory or specify another path)
    base_directory = '.'  # Change this to your desired base path

    # Create the file structure
    create_structure(structure, base_directory)

    print("File structure generation completed.")


if __name__ == "__main__":
    main()
