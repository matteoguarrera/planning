#@markdown ### **Installing pip packages**
#@markdown - Diffusion Model: [PyTorch](https://pytorch.org) & [HuggingFace diffusers](https://huggingface.co/docs/diffusers/index)
#@markdown - Dataset Loading: [Zarr](https://zarr.readthedocs.io/en/stable/) & numcodecs
#@markdown - Push-T Env: gym, pygame, pymunk & shapely
# !pip install gdown
# !python --version
# !pip3 uninstall cvxpy -y > /dev/null
# !pip3 install setuptools==65.5.0 > /dev/null
# # hack for gym==0.21.0 https://github.com/openai/gym/issues/3176
# !pip3 install torch==1.13.1 torchvision==0.14.1 diffusers==0.11.1 \
# scikit-image==0.19.3 scikit-video==1.1.11 zarr==2.12.0 numcodecs==0.10.2 \
# pygame==2.1.2 pymunk==6.2.1 gym==0.21.0 shapely==1.8.4 \
# &> /dev/null # mute output

import multiprocessing
from torch.utils.tensorboard import SummaryWriter

n_cores = multiprocessing.cpu_count()
# print(n_cores)


#@markdown ### **Imports**
# diffusion policy import
from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

# env import
# import gym
# from gym import spaces

# import gymnasium as gym
# from gymnasium import spaces
# import gymnasium as gym
import numpy as np

from datetime import datetime

# import pygame
# import pymunk
# import pymunk.pygame_util
# from pymunk.space_debug_draw_options import SpaceDebugColor
# from pymunk.vec2d import Vec2d
# import shapely.geometry as sg
# import cv2
# import skimage.transform as st
# from skvideo.io import vwrite
# from IPython.display import Video
import gdown
import os

import pickle
from importlib import reload  # Python 3.4+
import matplotlib.pyplot as plt


def set_reproductibility(seed=2023):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

##################### [No behavioral cloning] RL for new trajectory #######################
# from typing import Callable
# from stable_baselines3 import A2C, TD3
# from stable_baselines3 import SAC, PPO, A2C, DDPG
# from stable_baselines3.common.logger import configure
# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.env_util import make_vec_env

# def make_env(rank: int, seed: int = 0) -> Callable:
#     """
#     Utility function for multiprocessed env.
#     :param env_id: (str) the environment ID
#     :param num_env: (int) the number of environment you wish to have in subprocesses
#     :param seed: (int) the inital seed for RNG
#     :param rank: (int) index of the subprocess
#     :return: (Callable)
#     """
#
#     def _init():
#         env = PushTEnv()
#         env.reset(seed=seed + rank)
#         return env
#
#     set_random_seed(seed)
#     return _init






# ######################################## Helper functions ########################################
# positive_y_is_up: bool = False
# """Make increasing values of y point upwards.
#
# When True::
#
#     y
#     ^
#     |      . (3, 3)
#     |
#     |   . (2, 2)
#     |
#     +------ > x
#
# When False::
#
#     +------ > x
#     |
#     |   . (2, 2)
#     |
#     |      . (3, 3)
#     v
#     y
#
# """
#
# def light_color(color: SpaceDebugColor):
#     color = np.minimum(1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255]))
#     color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
#     return color
#
#
#
# def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
#     """Convenience method to convert pymunk coordinates to pygame surface
#     local coordinates.
#
#     Note that in case positive_y_is_up is False, this function wont actually do
#     anything except converting the point to integers.
#     """
#     if positive_y_is_up:
#         return round(p[0]), surface.get_height() - round(p[1])
#     else:
#         return round(p[0]), round(p[1])
#
#
#
#
# def pymunk_to_shapely(body, shapes):
#     geoms = list()
#     for shape in shapes:
#         if isinstance(shape, pymunk.shapes.Poly):
#             verts = [body.local_to_world(v) for v in shape.get_vertices()]
#             verts += [verts[0]]
#             geoms.append(sg.Polygon(verts))
#         else:
#             raise RuntimeError(f'Unsupported shape type {type(shape)}')
#     geom = sg.MultiPolygon(geoms)
#     return geom
