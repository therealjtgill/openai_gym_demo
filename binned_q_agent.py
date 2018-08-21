import gym
import numpy as np

class ObsTransformer(object):

  def __init__(self, env):

    NUM_BINS = 10
    INF_CAP = 2
    LEARNING_RATE = 0.001
    DISCOUNT = 0.9
    GREEDY_VAL = 0.2

    self.env = env
    self.obs_size = np.squeeze(env.observation_space.shape)
    obs_max = env.observation_space.high
    obs_min = env.observation_space.low
    #obs_max_capped = np.asarray([INF_CAP if h > INF_CAP else h for h in state_high])
    #obs_min_capped = np.asarray([-INF_CAP if l < -INF_CAP else l for l in state_low])
    obs_max_capped = np.asarray([2.4, 2, 0.4, 3.5])
    obs_min_capped = np.asarray([-2.4, -2, -0.4, -3.5])
    self.obs_max = obs_max_capped
    self.obs_min = obs_min_capped
    self.bins = [np.linspace(l, h, NUM_BINS) for l, h in zip(state_low_capped, state_high_capped)]

def obs_to_bin(obs, obs_min, obs_max, obs_size, bins):
  obs_capped = np.maximum(np.minimum(obs, obs_max), obs_min)
  bin_indices = []
  for i in range(obs_size):
    bin_indices.append(np.digitize(obs_capped[i], bins[i]))

  return tuple((np.asarray(bin_indices) - 1).astype(int))

if __name__ == "__main__":
  env = gym.make("CartPole-v0")
  old_obs = env.reset()
  print(old_obs)

  NUM_BINS = 10
  INF_CAP = 2
  LEARNING_RATE = 0.001
  DISCOUNT = 0.9
  GREEDY_VAL = 0.2

  state_size = np.squeeze(env.observation_space.shape)
  state_high = env.observation_space.high
  state_low = env.observation_space.low
  #state_high_capped = np.asarray([INF_CAP if h > INF_CAP else h for h in state_high])
  #state_low_capped = np.asarray([-INF_CAP if l < -INF_CAP else l for l in state_low])
  state_high_capped = np.asarray([2.4, 2, 0.4, 3.5])
  state_low_capped = np.asarray([-2.4, -2, -0.4, -3.5])
  bins = [np.linspace(l, h, NUM_BINS) for l, h in zip(state_low_capped, state_high_capped)]

  print(state_high_capped, state_low_capped)

  num_actions = 0
  while env.action_space.contains(num_actions):
    num_actions += 1

  print(env.action_space)

  print(state_size, num_actions)
  Q_shape = [NUM_BINS,]*int(state_size) + [num_actions,]
  print([NUM_BINS,]*int(state_size))
  print("Q_shape:", Q_shape)
  Q = np.random.random_sample(Q_shape)
  print("shape of Q:", Q.shape)

  run_times = []
  rewards = []
  for e in range(20000):
    old_obs = env.reset()
    total_reward = 0
    i = 0
    done = False
    while i < 1000:
      epsilon = np.random.rand()
      old_bin_obs = obs_to_bin(old_obs, state_low_capped, state_high_capped, state_size, bins)
      if epsilon <= 1.0/np.sqrt(e + 1):
        old_action = env.action_space.sample()
      else:
        # Pick optimal value from Q-table
        old_action = np.argmax(Q[old_bin_obs])
      new_obs, new_reward, done, _ = env.step(old_action)
      total_reward += new_reward
      if done and i < 199:
        new_reward = -300
      new_bin_obs = obs_to_bin(new_obs, state_low_capped, state_high_capped, state_size, bins)#find_bin(new_obs)
      q_old = Q[old_bin_obs][old_action]
      q_new = Q[new_bin_obs]
      Q[old_bin_obs][old_action] = (1 - LEARNING_RATE)*q_old + LEARNING_RATE*(new_reward + DISCOUNT*q_new.max())
      if done:
        if e % 100 == 0:
          print("reward:", total_reward)
          print("num timesteps:", i)
        run_times.append(i)
        rewards.append(total_reward)
        break
      old_obs = new_obs
      i += 1

  done = False
  old_obs = env.reset()
  while i < 1000:
    old_bin_obs = obs_to_bin(old_obs, state_low_capped, state_high_capped, state_size, bins)#find_bin(old_obs)
    action = np.argmax(Q[old_bin_obs])
    old_obs, new_reward, done, _ = env.step(action)
    env.render()
    if done:
      break

  import matplotlib.pyplot as plt
  plt.hist(run_times, bins='auto')
  plt.show()
  smoothed_rewards = [sum(rewards[i:i+100])/100 for i in range(len(rewards))]
  plt.plot(smoothed_rewards)
  plt.show()

  env.close()