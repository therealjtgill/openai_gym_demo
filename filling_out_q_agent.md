# Filling out `binned_q_agent.py`

The CartPole problem has four state (observation) features with continuous values. We're going to bin those up and do some Q Learning.

Don't hesitate to ask questions! This MD file is purposefully skimpy on details.

## Run it as it is:

Give it the ol'

`$ python binned_q_agent.py`

And see what happens. You should see a lot of output fly by that looks like this:

```
...
episode: 80
  reward: 10.0
episode: 90
  reward: 9.0
episode: 100
  reward: 9.0
episode: 110
  reward: 10.0
episode: 120
  reward: 9.0
episode: 130
  reward: 11.0
episode: 140
  reward: 9.0
...
```

Followed by a demo of your agent interacting with the environment executing what it's learned, and some helpful plots.

**What do these numbers mean?!**

`episode` is the number of episodes we've gone through so far
`reward` is the cumulative reward we've received at this particular episode (the sum of the rewards received at each timestep in this episode)

The higher the `reward`, the better our agent is doing.

The demo is for your entertainment.

The histogram plot shows you the count of the rewards your agent received as it learned (tall bars to the far right are good).

The line plot shows you the time-averaged reward your agent receives (ignore the last point; larger values are better).

## Make some modifications

You'll see several tags in the script that look like:

```
\# EDIT MEEEEE vvv
...some code here...
\# EDIT MEEEEE ^^^
```

These are for you to fill out. Try running the script and observing your agent's behavior as you fill out the tags.

### Tag #1

For the first tag, you're filling out maximum and minimum values of our binned observations. (Hint, use the output of the exploration notebook).

What to look for:

```
\# EDIT MEEEEE vvv
obs_max_capped = np.asarray([1.0, 1.0, 1.0, 1.0])
obs_min_capped = np.asarray([-1.0, -1.0, -1.0, -1.0])
\# EDIT MEEEEE ^^^
```

### Tag #2

The second tag is a little tougher. You're going to fill out the Q-table update algorithm. Be not afraid. Here's what the math looks like:

Q(s_1, a_1) <- Q(s_1, a_1) + learning_rate*(reward + gamma*argmax_over_a_2(Q(s_2, a_2)) - Q(s_1, a_1))

So all you really need to do is match up states and actions, and take a maximum. (Hint, np.asarray([1, 2, 3]).max = 3) (Hint, all of the variables that you need for this are passed to the method)

What to look for:

```
\# EDIT MEEEEE vvv
self.Q[old_state_t][old_action] += 0
\# EDIT MEEEEE ^^^
```

### Tag #3

A little bit easier, provide some default values. The first one is the number of bins to use for our discretized states. A good starting number is 10, just mind the number of bins you use: This will allocate an array of `NUM_BINS*NUM_BINS*NUM_BINS*NUM_BINS*2`.

`NUM_EPISODES` is the maximum number of episodes your agent will play as it attempts to learn how to maximize the reward it receives by taking actions.

What to look for:

```
\# EDIT MEEEEE vvv
NUM_BINS = 4
NUM_EPISODES = 200
\# EDIT MEEEEE ^^^
```
