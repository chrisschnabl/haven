# Analysis Directory

This directory contains all artifacts and code used to generate plots and analysis for the thesis research.

## Directory Structure

- `aws-analysis-*/` - Contains AWS-based analysis results and generated plots
- `local-analysis-*/` - Contains locally generated analysis results and plots
- `input_datasets/` - Source datasets used as inputs for the analysis
- `quantization_ablation_model/` - Local results from quantization experiments
- `remote_experiments/` - Data collected from remote experiment runs

## Key Files

- `local_analysis.py` - Script for analyzing quantization experiments
- `aws_analysis.py` - Script for analyzing enclave-based runs
- `shared_plot_helpers.py` - Common utilities for plot generation
- `pyproject.toml` - Project dependencies and configuration
- `uv.lock` - Lock file for dependency management