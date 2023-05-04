import platform
import torch
from parse import *


def get_model_parameters_for_diffusion_from_string(string):

    result = parse('pretrained/{}_arch{}_e{}_d{}_edim{}_ks{}_par{}_date{}', string)
    # system_name, arch, num_epochs, num_diffusion_iters, diffusion_step_embed_dim, kernel_size, num_param, date_time
    system_name, arch, _, num_diffusion_iters, diffusion_step_embed_dim, kernel_size, _, _ = result
    num_diffusion_iters = int(num_diffusion_iters)
    diffusion_step_embed_dim = int(diffusion_step_embed_dim)
    kernel_size = int(kernel_size)
    down_dims = [int(i) for i in arch.split('_')]

    params = get_model_parameters_for_diffusion()
    params['KERNEL_SIZE'] = kernel_size
    params['NUM_DIFFUSION_ITERS'] = num_diffusion_iters
    params['DIFFUSION_STEP_EMBEDDING_DIM'] = diffusion_step_embed_dim
    params['DOWN_DIMS'] = down_dims
    params['SHRINK'] = None

    return params


def get_model_parameters_for_diffusion():
    params = {
        # Weights and Biases - params
        'PROJECT_NAME': 'diffusion',
        'ENTITY': 'dl-282',
        # ID
        'ID': 0,
        # Hardware
        'DEVICE': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        # General
        'SHRINK': 1,
        'LOG_DIR': 'logs',
        'NUM_EPOCHS': 100,
        'DTYPE': torch.float32,
        # Model Architecture
        'KERNEL_SIZE': 5,
        'DOWN_DIMS': [256],
        'N_GROUPS': 2,
        # Diffusion
        'NUM_DIFFUSION_ITERS': 100,
        'DIFFUSION_STEP_EMBEDDING_DIM': 256,
        # Exponential Moving Average
        'EMA_POWER': 0.75,
        # Optimizer
        'OPTIMIZER': 'adamw',
        'LEARNING_RATE': 9e-5,
        'WEIGHT_DECAY': 3e-5,
        'COSINE_LR_NUM_WARMUP_STEPS': 500,
        'WANDB': False
    }
    params['IS_M1_ARCH'] = True if params['DEVICE'] == 'mps' else False
    params['BATCH_SIZE'] = 128 if params['IS_M1_ARCH'] else 256
    params['NUM_WORKERS'] = 2 if params['IS_M1_ARCH'] else 4
    params['ARCH'] = str(params['DOWN_DIMS'])[1:-1].replace(', ', '_')

    # Apply shrinkage to the model
    if params['SHRINK'] > 1:
        params['DOWN_DIMS'] = [d // params['SHRINK'] for d in params['DOWN_DIMS']]
        
    return params
        