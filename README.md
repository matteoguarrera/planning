# Final project CS282A: Diffusion Planning
[![GitHub stars](https://img.shields.io/github/stars/matteoguarrera/planning.svg)](https://GitHub.com/matteoguarrera/planning/stargazers/)  [![GitHub forks](https://img.shields.io/github/forks/matteoguarrera/planning.svg)](https://GitHub.com/matteoguarrera/planning/network/)  [![GitHub repo size in bytes](https://img.shields.io/github/repo-size/matteoguarrera/planning.svg)](https://github.com/matteoguarrera/planning) [![GitHub license](https://img.shields.io/github/license/matteoguarrera/planning.svg)](https://github.com/matteoguarrera/planning/blob/master/LICENSE)

**Authors**: Carlo Bosio<sup>1</sup>, Karim Elmaaroufi<sup>2</sup>, Margarita Geleta<sup>2</sup>, Matteo Guarrera<sup>2</sup>

<sup>1</sup> *UC Berkeley, Department of Mechanical Engineering*

<sup>2</sup> *UC Berkeley, Department of Electrical Engineering and Computer Science*

## Description
This repository contains the code for our project on diffusion models for drone motion planning.

## Usage
- [ ] Do we need to setup the environment?
- [ ] Do we install requirements?
```bash
>> pip3 uninstall cvxpy -y > /dev/null
>> pip3 install setuptools==65.5.0 > /dev/null
>> pip3 install torch==1.13.1 torchvision==0.14.1 diffusers==0.11.1 zarr==2.12.0 numcodecs==0.10.2
```

### Execution steps
**1. Download the dataset and the pretrained model from the v0.0.0 release**

Run all the cells from the `DatasetGenerator` notebook or make sure to download the dataset and put it in the 
`datasets` folder
You can specify there what datasets you want to generate.

**2. Launch training**

Run the `DiffusionTraining` notebook, if you have the computational power. We suggest to use the pretrained model and just do inference.
If you are using a M1 Mac, please execute the jupyter notebook with the following command to enable MPS:
```
PYTORCH_ENABLE_MPS_FALLBACK=1 jupyter notebook
```

**3. Inference**
Run the `DiffusionInference` notebook and visualize one trajectory. 


## Etc.
To do:
 - [ ] Make sure we haven't seen inference starting point,
   - I have already fixed seed for training, so this is just a double check. Plot the initial sampling starting condition (x,y),and the test set obs_0
   - np.random.seed() training and inference are disallined
 - [ ] Accuracy measured as number fineshed trajectories within n_step / total trajectories
 - [ ] Comment the load dataset functions
 - [ ] plug the diffusion in the loop
   - Option 1, actual train a diffusion to output the right acceleration, given the drone model
   - Option 2, train a diffusion to predict the next observation (we might use the model we already have)
     - For this option we are predicting the next position based on a fake model of drone. 
     - We use the 2D dataset and we let the drone follow the waypoint produced by the 8 action taken, 
and we produce the next 8 observation based on those. Those observations become the waypoint that the drone has to follow.
 - [ ] Fix dataset generator for drone, and re run experiment for drone only
   - Reduce the length of gdown decrease the parameters but doesn't increase the speed, same performances
   - Maybe introduce some plot for dataset generator,
   - Plot training loss and standard deviation along with hyperparameters.
