import numpy as np
from gym import spaces
from scipy.spatial.transform import Rotation as R

from DATT.learning.base_env import BaseQuadsimEnv
from DATT.configuration.configuration import AllConfig
from DATT.quadsim.lineref import LineRef


class TrajectoryEnv(BaseQuadsimEnv):
  """
  Quadsim environment that also contains a (x, y, z) reference trajectory. 
  """
  def __init__(self, config: AllConfig, ref=None, seed=None, fixed_seed=False, **kwargs):
    self.time_horizon = config.policy_config.time_horizon
    self.fb_term = config.policy_config.fb_term
    self.include_assumed_params = config.sim_config.sampler.sample_param(config.sim_config.include_env_info)
    print('TIME HORIZON: ', self.time_horizon)
    print('USING FB: ', self.fb_term)
    self.seed = seed
    if self.seed is None:
      self.seed = np.random.randint(0, 1000000)
    self.reset_count = 0
    self.reset_freq = config.training_config.reset_freq
    self.reset_thresh = config.training_config.reset_thresh

    if ref is None:
        self.ref = LineRef(D=1.0, altitude=0.0, period=1)
    else:
      self.ref = ref.ref(config.ref_config, seed=self.seed, env_diff_seed=config.training_config.env_diff_seed, fixed_seed=fixed_seed)
    self.dt = 0.02

    state_dim = 13  # Assuming state includes position (3), velocity (3), orientation (4), angular velocity (3)
    self.state_history = np.zeros((10, state_dim))

    super().__init__(config=config, **kwargs)

    #self.ref = CircleRef(rad=1, altitude=0.0)
    self.action_space = spaces.Box(low=np.array([-9.8, -20, -20, -2]), high=np.array([30, 20, 20, 2]))

    # MODIFIED:
    # This will normally be run with the fb_term and include_assumed_params true. 

    if self.fb_term:
      if self.include_assumed_params:
        self.all_mins = np.r_[self.all_mins, -50 * np.ones(3 * (self.time_horizon + 1) + 10*state_dim + 10)]
        self.all_maxes = np.r_[self.all_maxes, 50*np.ones(3 * (self.time_horizon + 1) + 10*state_dim + 10)]
      else:
        self.all_mins = np.r_[self.all_mins, -50 * np.ones(3 * (self.time_horizon + 1) + 10*state_dim)]
        self.all_maxes = np.r_[self.all_maxes, 50*np.ones(3 * (self.time_horizon + 1) + 10*state_dim)]
    else:
      self.all_mins = np.r_[self.all_mins, -50 * np.ones(3 * (self.time_horizon) + 10*state_dim)]
      self.all_maxes = np.r_[self.all_maxes, 50*np.ones(3 * (self.time_horizon) + 10*state_dim)]

    if self.include_assumed_params:
      self.observation_shape = (self.observation_shape[0] + 10 + 10*state_dim + 10,)
    else:
      self.observation_shape = (self.observation_shape[0] + 10*state_dim+10,) 
    self.observation_space = spaces.Box(low=self.all_mins, high=self.all_maxes)
    
  def reset(self, state=None):
    self.reset_count += 1      
    self.state_history = np.zeros_like(self.state_history)

    if (self.reset_freq > 0 and self.reset_count % self.reset_freq == 0) or (self.reset_count > self.reset_thresh):
      try:
        self.ref.reset()
      except AttributeError:
        pass
    return super().reset(state)

  def obs(self, state):
    obs_ = super().obs(state)
    rot = R.from_quat(obs_[6:10])

    current_state = np.hstack([state.pos, state.vel, state.rot.as_quat(), state.ang])
    self.state_history = np.roll(self.state_history, -1, axis=0)
    self.state_history[-1] = current_state

    history_flat = self.state_history.flatten()

    if self.body_frame:
      fb = obs_[0:3] - rot.inv().apply(self.ref.pos(self.t))
      if self.fb_term:
        obs_ = np.hstack([obs_, fb, history_flat] + [obs_[0:3] - rot.inv().apply(self.ref.pos(self.t + 3 * i * self.dt)) for i in range(self.time_horizon)])
      else:
        velquat = obs_[3:]
        obs_ = np.hstack([fb, velquat, history_flat] + [obs_[0:3] - rot.inv().apply(self.ref.pos(self.t + 3 * i * self.dt)) for i in range(self.time_horizon)])
    else:
      fb = obs_[0:3] - self.ref.pos(self.t)
      if self.fb_term:
        obs_ = np.hstack([obs_, fb, history_flat] + [self.ref.pos(self.t + 3 * i * self.dt) for i in range(self.time_horizon)])
      else:
        velquat = obs_[3:]
        obs_ = np.hstack([fb, velquat, history_flat] + [self.ref.pos(self.t + 3 * i * self.dt) for i in range(self.time_horizon)])
    
    if self.include_assumed_params: # MODIFIED to allow RMA Phase 1 Policies to get the true env. parameters
      obs_ = np.hstack([obs_, self.get_env_params()])

    return obs_

  def reward(self, state, action):
    # pos_scalar = 0.1
    # rot_scalar = 0.02
    # vel_scalar = 0.01
    # act_scalar = 0.001

    # yaw = state.rot.as_euler('ZYX')[0]
    # yawcost = rot_scalar * np.linalg.norm(yaw-self.ref.yaw(self.t))**2
    # poscost = pos_scalar * np.linalg.norm(state.pos - self.ref.pos(self.t))**2
    # velcost = vel_scalar * np.linalg.norm(state.vel)**2

    # actcost = act_scalar * np.linalg.norm(action)**2

    # cost = yawcost + poscost + velcost + actcost

    yaw = state.rot.as_euler('ZYX')[0]
    yawcost = 0.5 * min(abs(self.ref.yaw(self.t) - yaw), abs(self.ref.yaw(self.t) - yaw))
    poscost = np.linalg.norm(state.pos - self.ref.pos(self.t))#min(np.linalg.norm(state.pos), 1.0)
    velcost = 0.1 * min(np.linalg.norm(state.vel), 1.0)

    # ucost = 0.2 * abs(action[0]) + 0.1 * np.linalg.norm(action[1:])

    cost = yawcost + poscost + velcost

    return -cost
  
  def get_env_params(self):
    return np.hstack([self.assumed_mass, self.assumed_I.flatten()])
