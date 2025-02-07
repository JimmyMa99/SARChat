# Train

## Environment Setup and Running Instructions

### Requirements
All experiments are based on the [Swift framework](https://github.com/modelscope/ms-swift), so the environment needs to be configured first.

#### 1. Install Swift 
```bash
# Create a new Python environment
conda create -n your_env_name python=3.10  # Python 3.10 is recommended
conda activate your_env_name

# Install Swift framework
pip install 'ms-swift[all]' -U
```

#### 2. Download Model Weights
Please refer to [Swift Supported Multimodal Models](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.html#id4) to download the required model weights. For continued training of our models, please download the corresponding model weights from [README](../README.md) and fill in the script path.

#### 3. Configure Paths
In the provided shell script, please fill in the following paths:
- Model weights path  
- Dataset path
- Output path

#### 4. Run Experiments
```bash
# Activate environment
conda activate your_env_name

# Run corresponding experiment script 
bash ./train/swift_run/run_experiment.sh  # Replace run_experiment.sh with the actual script name
```

### Notes
- Ensure all dependencies are installed correctly
- Verify the integrity of model weight files
- Check path configurations are correct

