# QKoopman

**Authors**: Baoyang Zhang, Zhen Lu, Yaomin Zhao, Yue Yang  
**Paper**: [arXiv:2507.21890](https://arxiv.org/abs/2507.21890)  

QKoopman is an experimental research project exploring the potential of quantum-accelerated nonlinear dynamics simulation through a novel data-driven quantum Koopman method that combines deep learning for global linearization with quantum algorithms for unitary evolution.

## Installation

To set up the QKoopman environment:

```bash
# Create and activate a conda environment
conda create -n qk python=3.10.12
conda activate qk

# Install required dependencies
pip install -r requirements.txt
```

## Project organization

This repository contains implementations for three fundamental nonlinear dynamics cases:

- **Reaction-diffusion system** (`gray/`)  

- **2D turbulence** (`kol/`)  

- **Shear flow** (`shear/`)  

Post-processing tools are available in `kpo/`.

## Getting Started


### Training with DeepSpeed

We utilize [DeepSpeed](https://www.deepspeed.ai/) for distributed data-parallel training. 

The training process can be launched using the following shell script:

```bash
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

deepspeed --num_gpus ${NUM_GPUS} \
          gd_h4.py  \
          --deepspeed \
          --deepspeed_config ds_config.json \
          > run.log 2>&1
```

### Testing & Reproduction

To evaluate the models and reproduce results from our paper:

1. **Download the dataset**  
   The data for this project is hosted on Hugging Face Datasets:  
   [![Dataset](https://img.shields.io/badge/🤗%20Datasets-QKoopman-blue)](https://huggingface.co/datasets/YYgroup/QKoopman)

2. **Run visualization notebook**  
   Execute our Jupyter notebook for generating figures:
   ```bash
   jupyter notebook ./kpo/draw.ipynb
    ```

## Citation

If you use our code, please cite:

```bibtex
@misc{zhang2025datadrivenquantumkoopmanmethod,
      title={Data-driven quantum {Koopman} method for simulating nonlinear dynamics}, 
      author={Baoyang Zhang and Zhen Lu and Yaomin Zhao and Yue Yang},
      year={2025},
      eprint={2507.21890},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2507.21890}, 
}
```