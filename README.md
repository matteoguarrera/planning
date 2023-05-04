# Final project CS282A: Diffusion Planning
[![GitHub stars](https://img.shields.io/github/stars/matteoguarrera/planning.svg)](https://GitHub.com/matteoguarrera/planning/stargazers/)  [![GitHub forks](https://img.shields.io/github/forks/matteoguarrera/planning.svg)](https://GitHub.com/matteoguarrera/planning/network/)  [![GitHub repo size in bytes](https://img.shields.io/github/repo-size/matteoguarrera/planning.svg)](https://github.com/matteoguarrera/planning) [![GitHub contributors](https://img.shields.io/github/contributors/margaritageleta/vesper-tech-debt.svg)](https://GitHub.com/matteoguarrera/planning/graphs/contributors/) [![GitHub license](https://img.shields.io/github/license/matteoguarrera/planning.svg)](https://github.com/matteoguarrera/planning/blob/master/LICENSE)

**Authors**: Carlo Bosio<sup>1</sup>, Karim Elmaaroufi<sup>2</sup>, Margarita Geleta<sup>2</sup>, Matteo Guarrera<sup>2</sup>

<sup>1</sup> *UC Berkeley, Department of Mechanical Engineering*

<sup>2</sup> *UC Berkeley, Department of Electrical Engineering and Computer Science*

## Description
This repository contains the code for our project on diffusion models for drone motion planning.

## Usage
```bash
pip install -r requirements.txt
```

### Execution steps
**1. Download the pretrained models from the v0.0.1 release**

**2. Launch training**
Run the following, if you have the computational power.
```bash
  >> python diffusion.py
```
If not, we suggest using the pretrained model and just do inference.

If you are using a M1 Mac, please execute any jupyter notebook with the following command to enable MPS:
```
PYTORCH_ENABLE_MPS_FALLBACK=1 jupyter notebook
```

**3. Inference**
Run the `DiffusionInference` notebook and visualize one trajectory. 

## Simulator

For the simulator installation please follow [this link.](https://cyberbotics.com/doc/guide/installation-procedure)
