#!/bin/bash

# Activate environment
conda activate 215a

# Install libraries required in environment from yaml file
conda env update --file environment.yaml

# Run the script for data cleaning
python clean.py

# Run the analysis script
python analysis.py


