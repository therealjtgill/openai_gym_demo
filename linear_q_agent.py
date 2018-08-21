import gym
import numpy as np

if __name__ == "__main__":
  env = gym.make("CartPole-v0")
  env.reset()

  state_size = env.observation_space.shape[0]
  action_size = env.action_space.shape[0]

  W = np.random.rand((state_size, action_size))
  b = np.random.rand((action_size,))

  done = False
  i = 0

  while i < 1000 and not done:
    
    env.step()