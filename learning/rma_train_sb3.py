import importlib.util
import os
import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from DATT.configuration.configuration import AllConfig, RefConfiguration

from DATT.learning.utils.feedforward_feature_extractor import \
    FeedforwardFeaturesExtractor

from DATT.learning.configs import *
from DATT.learning.tasks import DroneTask
from DATT.refs import TrajectoryRef

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import make_proba_distribution
from stable_baselines3.common.utils import get_device
from gym.spaces import Box

from custom_policies import RMAFeaturesExtractor, RMAEncoder, RMAPolicy
from custom_policies import RPGFeaturesExtractor, RPGPolicy


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-t', '--task', dest='task', 
        type=DroneTask, default=DroneTask.HOVER,
        help='Name of the task. Defined in ./tasks/ Default: hover'
    )
    parser.add_argument('-a', '--algo', dest='algo', 
        type=RLAlgo, default=RLAlgo.PPO,
        help='Name of the RL algorithm to train the policy. Default: ppo'
    )
    parser.add_argument('-c', '--config', dest='config',
        default='default_hover.py',
        help='Name of the configuration .py file. Default: default_hover.py'    
    )
    parser.add_argument('-n', '--name', dest='name', 
        default=None,
        help='Name of the policy to train. If such policy already exists in ./saved_policies/, continues training it.'
    )
    parser.add_argument('-d', '--log-dir', dest='log_dir',
        default=None,
        help='The directory to log training info to. Can run tensorboard from this directory to view.'   
    )
    parser.add_argument('-ts', '--timesteps', dest='timesteps',
        type=int, default=1e6,
        help='Number of timesteps to train for. Default: 1 million'    
    )
    parser.add_argument('-sd', '--save-data', dest='save_data',
        type=bool, default=False,
        help='bool, whether to save state transition data for offline supervised learning, etc.'
    )
    parser.add_argument('-dd', '--data-dir', dest='data_dir',
        default=None,
        help='Directory to save data to, if saving data.'
    )
    parser.add_argument('-ch', '--checkpoint', dest='checkpoint',
        type=bool, default=False,
        help='Whether to save checkpoints.'
    )
    parser.add_argument('-de', '--device', dest='device',
        type=int, default=0,
        help='GPU ID to use.'
    )

    parser.add_argument("--model", type=str, required=True, help="Model to use (RMA or RPG)")
    
    parser.add_argument('--n-envs', type=int, help='How many "parallel" environments to run', default=10)
    parser.add_argument('-r', '--ref', dest='ref', type=TrajectoryRef, default=TrajectoryRef.LINE_REF)
    parser.add_argument('--seed', dest='seed', type=int, default=None,
        help='Seed to use for randomizing reference trajectories during training.'
    )

    args = parser.parse_args()

    return args

def find_default_name_num(dir, prefix):
    seen_nums = set()
    for name in os.listdir(dir):
        if name.startswith(f'{prefix}_'):
            try:
                num = int(name[len(prefix) + 1:])
            except ValueError:
                pass
            else:
                seen_nums.add(num)
    
    num = 0
    while num in seen_nums:
        num += 1

    return f'{prefix}_{num}'


# def import_config(config_filename):
#     spec = importlib.util.spec_from_file_location("config", CONFIG_DIR / config_filename)
    
#     config_module = importlib.util.module_from_spec(spec)
#     sys.modules['config'] = config_module
#     spec.loader.exec_module(config_module)
#     try:
#         return config_module.config
#     except AttributeError:
#         raise ValueError(f'Config file {config_filename} must define a config object named `config`.')

def train():
    args = parse_args()
    
    task: DroneTask = args.task
    policy_name = args.name
    ref = args.ref
    log_dir = args.log_dir
    ts = args.timesteps
    algo = args.algo
    config_filename = args.config
    save_data = args.save_data
    checkpoint = args.checkpoint
    data_dir = args.data_dir
    device = args.device
    seed = args.seed
    n_envs = args.n_envs

    if policy_name is None:
        policy_name = f'{task.value}_{algo.value}_policy'

    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR / f'{policy_name}_logs'
        log_dir.mkdir(exist_ok=True, parents=True)
    if not log_dir.exists():
        raise FileNotFoundError(f'{log_dir} does not exist')

    if not (CONFIG_DIR / config_filename).exists():
        raise FileNotFoundError(f'{config_filename} is not a valid config file')

    if save_data and data_dir is None:
        data_dir = DEFAULT_DATA_DIR / policy_name
        data_dir.mkdir(exist_ok=True, parents=True)
    if save_data:
        if not data_dir.exists():
            raise FileNotFoundError(f'{data_dir} does not exist')
        data_filename = data_dir / find_default_name_num(data_dir, f'{policy_name}_data')
        print(f'Saving data to {data_filename}')
        with open(data_filename, 'w+') as f:
            pass
    else:
        data_filename = None

    config: AllConfig = import_config(config_filename)
    env_kwargs={
        'config': config,
        'save_data': save_data, 
        'data_file': data_filename,
    }

    if task.is_trajectory():
        env_kwargs['ref'] = ref
        if seed is not None:
            env_kwargs['seed'] = seed
        else:
            env_kwargs['seed'] = np.random.randint(0, 100000)

    env_class = task.env()

    if issubclass(env_class, VecEnv):
      env = VecMonitor(env_class(n_envs))
    else:
      env = make_vec_env(env_class, n_envs=n_envs, env_kwargs=env_kwargs)

    print(env.reset().shape)

    algo_class = algo.algo_class()


    if not (SAVED_POLICY_DIR / f'{policy_name}.zip').exists():
        print('Training new policy!')
        if args.model == "RMA" or args.model == "rma":
            encoder_net_arch = [64, 64]
            encoder_output_dim = 5
            encoder_input_dim = 10
            policy_net_arch = [dict(pi=[64, 64, 64], vf=[64, 64, 64])]
            policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=policy_net_arch,
                        encoder_input_dim = encoder_input_dim,
                        encoder_output_dim = encoder_output_dim, 
                        encoder_network_architecture= encoder_net_arch
            )

            kwargs = {}
            if issubclass(algo_class, OffPolicyAlgorithm):
                kwargs['train_freq'] = (5000, 'step')

            policy_network_type = 'RMAPolicy'
            print(f'Using policy network type: {policy_network_type}')
            
            policy: BaseAlgorithm = algo_class(
                RMAPolicy, 
                env, 
                tensorboard_log=log_dir, 
                policy_kwargs=policy_kwargs,
                device=f'cuda:{device}',
                verbose=0,
                **kwargs
            )
        elif args.model == "RPG" or args.model == "rpg":
            policy_net_arch = [dict(pi=[128, 128, 128], vf=[128, 128, 128])]
            policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=policy_net_arch,
            )

            kwargs = {}
            if issubclass(algo_class, OffPolicyAlgorithm):
                kwargs['train_freq'] = (5000, 'step')

            policy_network_type = 'RPGPolicy'
            print(f'Using policy network type: {policy_network_type}')

            policy: BaseAlgorithm = algo_class(
                RPGPolicy, 
                env, 
                tensorboard_log=log_dir, 
                policy_kwargs=policy_kwargs,
                device=f'cuda:{device}',
                verbose=0,
                **kwargs
            )
    else:
        policy: BaseAlgorithm = algo_class.load(SAVED_POLICY_DIR / f'{policy_name}.zip', env)
        print('CONTINUING TRAINING!')

    if checkpoint:
        checkpoint_callback = CheckpointCallback(
            save_freq=250000,
            save_path=SAVED_POLICY_DIR,
            name_prefix=policy_name
        )
    else:
        checkpoint_callback = None


    print(env.reset().shape)

    policy.learn(total_timesteps=ts, progress_bar=True, callback=checkpoint_callback)
    policy.save(SAVED_POLICY_DIR / policy_name)

if __name__ == "__main__":
    train()
