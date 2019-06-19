import gym
import gym.spaces
import numpy as np
import random

from atari_wrappers import make_atari, wrap_deepmind

class TransposeWrapper(gym.ObservationWrapper):
  def observation(self, observation):
    return np.transpose(np.array(observation), axes=(2,0,1))

class NoRwdResetEnv(gym.Wrapper):
  def __init__(self, env, no_reward_thres):
    """Reset the environment if no reward received in N steps
    """
    gym.Wrapper.__init__(self, env)
    self.no_reward_thres = no_reward_thres
    self.no_reward_step = 0

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    if reward == 0.0:
      self.no_reward_step += 1
    else:
      self.no_reward_step = 0
    if self.no_reward_step > self.no_reward_thres:
      done = True
    return obs, reward, done, info

  def reset(self, **kwargs):
    obs = self.env.reset(**kwargs)
    self.no_reward_step = 0
    return obs

def make_final(env_id, episode_life=True, clip_rewards=True, frame_stack=True, scale=True):
  env = wrap_deepmind(make_atari(env_id), episode_life, clip_rewards, frame_stack, scale)
  env = TransposeWrapper(env)
  env = NoRwdResetEnv(env, no_reward_thres = 1000)
  return env