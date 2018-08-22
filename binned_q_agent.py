import gym
import numpy as np

class ObsTransformer(object):

  def __init__(self, env, num_bins):

    INF_CAP = 2

    self.env = env
    self.obs_size = np.squeeze(env.observation_space.shape)
    self.num_bins = num_bins
    obs_max = env.observation_space.high
    obs_min = env.observation_space.low

    # EDIT MEEEEE vvv
    obs_max_capped = np.asarray([1.0, 1.0, 1.0, 1.0])
    obs_min_capped = np.asarray([-1.0, -1.0, -1.0, -1.0])
    # EDIT MEEEEE ^^^

    self.obs_max = obs_max_capped
    self.obs_min = obs_min_capped
    self.bins = [np.linspace(l, h, num_bins) for l, h in zip(obs_min_capped, obs_max_capped)]

  def obs_to_bin(self, obs):
    obs_capped = np.maximum(np.minimum(obs, self.obs_max), self.obs_min)
    bin_indices = []
    for i in range(self.obs_size):
      bin_indices.append(np.digitize(obs_capped[i], self.bins[i]))

    return tuple((np.asarray(bin_indices) - 1).astype(int))

class QAgent(object):

  def __init__(self, num_actions, state_size, num_bins, gamma, learning_rate, env):

    self.transformer = ObsTransformer(env, num_bins)
    self.gamma = gamma
    self.learning_rate = learning_rate
    self.Q = np.zeros([int(num_bins),]*int(state_size) + [num_actions,])

  def update(self, reward, old_state, old_action, new_state):

    old_state_t = self.transformer.obs_to_bin(old_state)
    new_state_t = self.transformer.obs_to_bin(new_state)

    # EDIT MEEEEE vvv
    self.Q[old_state_t][old_action] += reward
    # EDIT MEEEEE ^^^

  def predict(self, state):
    state_t = self.transformer.obs_to_bin(state)
    return np.argmax(self.Q[state_t])

if __name__ == "__main__":
  env = gym.make("CartPole-v0")

  num_actions = 0
  while env.action_space.contains(num_actions):
    num_actions += 1

  # Given values
  LEARNING_RATE = 0.001
  GAMMA = 0.9
  MAX_TIMESTEPS = 250

  # EDIT MEEEEE vvv
  NUM_BINS = 4
  NUM_EPISODES = 200
  # EDIT MEEEEE ^^^

  q_agent = QAgent(num_actions, env.observation_space.shape[0], NUM_BINS, GAMMA, LEARNING_RATE, env)

  #################
  # Train Q Agent #
  #################
  run_times = []
  rewards = []
  # Loop over some number of episodes
  for e in range(NUM_EPISODES):
    old_obs = env.reset()
    total_reward = 0
    i = 0
    done = False
    # Take steps in the environment
    while i < MAX_TIMESTEPS:
      epsilon = np.random.rand()
      if epsilon <= 1.0/np.sqrt(e + 1):
        old_action = env.action_space.sample()
      else:
        # Pick optimal value from Q-table
        old_action = q_agent.predict(old_obs)
      new_obs, new_reward, done, _ = env.step(old_action)
      total_reward += new_reward
      if done and i < 199:
        new_reward = -300
      q_agent.update(new_reward, old_obs, old_action, new_obs)
      if done:
        if e % 10 == 0:
          print("episode:", e)
          print("  reward:", total_reward)
        run_times.append(i)
        rewards.append(total_reward)
        break
      old_obs = new_obs
      i += 1

  ###############
  # Run Q Agent #
  ###############
  done = False
  old_obs = env.reset()
  while i < MAX_TIMESTEPS:
    action = q_agent.predict(old_obs)
    old_obs, new_reward, done, _ = env.step(action)
    env.render()
    if done:
      break

  ##########################
  # Show some useful plots #
  ##########################
  import matplotlib.pyplot as plt
  plt.hist(run_times, bins='auto')
  plt.title("Histogram of cumulative rewards received")
  plt.show()
  smoothed_rewards = [sum(rewards[i:i+100])/100 for i in range(len(rewards))]
  plt.plot(smoothed_rewards)
  plt.title("Smoothed cumulative rewards as a function of time")
  plt.show()

  env.close()