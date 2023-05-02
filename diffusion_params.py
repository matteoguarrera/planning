import platform
import torch

def get_model_parameters_for_diffusion():
    params = {
        # Weights and Biases - params
        'PROJECT_NAME': 'diffusion',
        'ENTITY': 'dl-282',
        # ID
        'ID' : 0,
        # Hardware
        'DEVICE': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        # General
        'SHRINK' : 1,
        'LOG_DIR': 'logs',
        'NUM_EPOCHS': 10,
        'DTYPE' : torch.float32,
        # Model Architecture
        'KERNEL_SIZE': 5,
        'DOWN_DIMS' : [1024],
        'N_GROUPS': 2,
        # Diffusion
        'NUM_DIFFUSION_ITERS': 50,
        'DIFFUSION_STEP_EMBEDDING_DIM': 256,
        # Exponential Moving Average
        'EMA_POWER' : 0.75,
        # Optimizer
        'OPTIMIZER': 'adamw',
        'LEARNING_RATE': 9e-5,
        'WEIGHT_DECAY': 3e-5,
        'COSINE_LR_NUM_WARMUP_STEPS': 500  
    }
    params['IS_M1_ARCH'] = True if params['DEVICE'] == 'mps' else False
    params['BATCH_SIZE'] = 128 if params['IS_M1_ARCH'] else 256
    params['NUM_WORKERS'] = 2 if params['IS_M1_ARCH'] else 4

    # Apply shrinkage to the model
    if params['SHRINK'] > 1:
        params['DOWN_DIMS'] = [d // params['SHRINK'] for d in params['DOWN_DIMS']]
        
    return params
        