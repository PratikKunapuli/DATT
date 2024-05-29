import torch
import torch.nn as nn
import numpy as np

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import make_proba_distribution
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import create_mlp

from gym.spaces import Box

class RMAFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, encoder, encoder_input_dim):
        # Example: concatenating encoded features with original observation
        # Adjust the feature_dim accordingly
        super(RMAFeaturesExtractor, self).__init__(observation_space, features_dim=encoder.output_dim + observation_space.shape[0])
        self.encoder = encoder
        self.encoder_input_dim = encoder_input_dim

    def forward(self, observations):
        # last 10 elements are the encoder input
        obs, encoder_input = observations[:, :-self.encoder_input_dim], observations[:, -self.encoder_input_dim:]
        encoding = self.encoder(encoder_input)
        return torch.cat((obs, encoding), dim=1)
    

class RMAEncoder(nn.Module):
    def __init__(self, input_dim, network_architecture, output_dim, activation_fn):
        super(RMAEncoder, self).__init__()
        self.input_dim = input_dim
        self.network_architecture = network_architecture
        self.output_dim = output_dim
        self.activation_fn = activation_fn

        self.encoder_layers = []

        for i in range(len(network_architecture) - 1):
            if i == 0:
                self.encoder_layers.append(nn.Linear(input_dim, network_architecture[i]))
            else:
                self.encoder_layers.append(nn.Linear(network_architecture[i-1], network_architecture[i]))
            self.encoder_layers.append(activation_fn())
        self.encoder_layers.append(nn.Linear(network_architecture[-1], output_dim))

        self.encoder = nn.Sequential(*self.encoder_layers)
    
    def forward(self, x):
        return self.encoder(x)
    
class RMAPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, encoder_input_dim, encoder_network_architecture, encoder_output_dim,
                 net_arch=None, activation_fn=nn.Tanh, *args, **kwargs):
        modified_observation_space = Box(low=np.concatenate([observation_space.low[:-encoder_input_dim], np.full(encoder_output_dim, -np.inf)]), high=np.concatenate([observation_space.high[:-encoder_input_dim], np.full(encoder_output_dim, np.inf)]))
        super(RMAPolicy, self).__init__(modified_observation_space, action_space, lr_schedule, 
                                        net_arch=net_arch, 
                                           activation_fn=activation_fn, *args, **kwargs)
        self.encoder = RMAEncoder(encoder_input_dim, encoder_network_architecture, encoder_output_dim, activation_fn)
        self.features_extractor = RMAFeaturesExtractor(observation_space, self.encoder, encoder_input_dim)

        
        
        self.encoder.to(self.device)
        # Re-initialize the actor and critic with the correct input dimensions
        self.action_dist = make_proba_distribution(action_space)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1).to(self.device)
        
    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        values = self.value_net(latent_vf)

        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()

        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_constructor_parameters(self):
        data = super(RMAPolicy, self)._get_constructor_parameters()
        data.update(dict(
            encoder_input_dim=self.encoder.input_dim,
            encoder_network_architecture=self.encoder.network_architecture,
            encoder_output_dim=self.encoder.output_dim
        ))
        return data

    def predict(self, observation, state=None, mask=None, deterministic=False, features_included = False):
        observation = torch.as_tensor(observation).float().to(self.device)
        if not features_included:
            features = self.extract_features(observation.unsqueeze(0))
        else:
            features = observation.unsqueeze(0)
        latent_pi, _ = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        action = action.cpu().detach().numpy().squeeze()
        return action, state
    
    def rollout_prediction(self, observation):
        observation = torch.as_tensor(observation).float().to(self.device)
        features = observation.unsqueeze(0)
        latent_pi, _ = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        action = distribution.mode()
        action = action.cpu().detach().numpy().squeeze()
        return action

class RPGFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        # Example: concatenating encoded features with original observation
        # Adjust the feature_dim accordingly
        super(RPGFeaturesExtractor, self).__init__(observation_space, features_dim)
        print("Observation Space: ", observation_space.shape)
        

    def forward(self, observations):
        # last 10 elements are the privaleged info
        obs_without_params = observations[:, :-10]
        assumed_params = observations[:, -10:]
        # Concatenate the assumed parameters to the extracted features
        return torch.cat((obs_without_params, assumed_params), dim=1)

class RPGPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(RPGPolicy, self).__init__(
            *args,
            **kwargs
        )
        self.policy_net, self.value_net = self._build_custom_networks(kwargs['net_arch'])
        self.features_extractor = RPGFeaturesExtractor(self.observation_space, 128)

    def _build_custom_networks(self, net_arch):
        # Extract the feature dimensions
        policy_feature_dim = self.observation_space.shape[0] - 10
        value_feature_dim = self.observation_space.shape[0]

        # Extract the architecture from net_arch
        pi_arch = net_arch[0]['pi']
        vf_arch = net_arch[0]['vf']

        # Create separate networks for policy and value functions
        policy_net = nn.Sequential(
            *create_mlp(policy_feature_dim, 128, net_arch=pi_arch),
        )

        value_net = nn.Sequential(
            *create_mlp(value_feature_dim, 1, net_arch=vf_arch),
        )

        return policy_net, value_net

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        policy_features = features[:, :-10]
        value_features = features

        policy_latent = self.policy_net(policy_features)

        distribution = self._get_action_dist_from_latent(policy_latent)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        value = self.value_net(value_features)
        return actions, value, log_prob

    def _get_latent(self, obs):
        features = self.extract_features(obs)
        policy_features = features[:, :-10]
        value_features = features
        policy_latent = self.policy_net(policy_features)
        return policy_latent, value_features

    def evaluate_actions(self, obs, actions):
        policy_latent, value_features = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(policy_latent)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        value = self.value_net(value_features)
        return value, log_prob, entropy

    def predict_values(self, obs):
        value_features = self.extract_features(obs)
        return self.value_net(value_features)
    
    def predict(self, observation, state=None, mask=None, deterministic=False):
        observation = torch.as_tensor(observation).float().to(self.device)
        features = self.extract_features(observation.unsqueeze(0))
        policy_features = features[:, :-10]
        latent_pi = self.policy_net(policy_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        action = action.cpu().detach().numpy().flatten()
        return action, state