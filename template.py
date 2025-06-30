import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    ".github/workflows/.gitkeep",  # For CI/CD workflows
    "data/raw/.gitkeep",           # Per assessment doc 
    "data/processed/.gitkeep",     # Per assessment doc 
    "notebooks/EDA.ipynb",         # Per assessment doc 
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_preprocessing.py", # Per assessment doc 
    f"src/{project_name}/components/feature_engineering.py",# Per assessment doc 
    f"src/{project_name}/components/model_dev.py",          # Corresponds to model.py 
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",   # Corresponds to train.py 
    "models/.gitkeep",             # For storing trained models 
    "reports/.gitkeep",            # For storing reports and figures 
    "main.py",                     # A main script to run the training pipeline
    "requirements.txt",            # For project dependencies [cite: 73]
    "README.md"                    # Project overview and setup instructions [cite: 73]
]


# Loop through the list to create the files and directories
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create the directory if it doesn't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # Create the file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # Create an empty file
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")