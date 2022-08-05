# # Training script
#
# ## Imports

# +
from unityagents import UnityEnvironment
import numpy as np

import matplotlib.pyplot as plt

# %matplotlib inline

import torch

# -

# ## Initializing environment

env = UnityEnvironment(file_name="Crawler.app", no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Value for random seed
seed = 42

# ### Checking environment and setting utility variables

# +
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print(f"number of agents = {num_agents}")
# size of each action
action_size = brain.vector_action_space_size
print(f"action size: {action_size}")
# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print(f"state size: {state_size}")
# -

# ## Instantiate agent

# +
from agent import Agent

agent = Agent(state_size, action_size, seed)
# -

# ## Training

# +
from collections import namedtuple, deque


def ddpg(n_episodes=100000, window_len=100, goal=30, print_every=1):
    original_goal = goal
    scores_deque = deque(maxlen=window_len)
    scores = []
    avg_scores = []
    for i_episode in range(1, n_episodes + 1):
        if i_episode % 1000 == 0:
            env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
        else:
            env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        # agent.reset()
        score = np.zeros(num_agents)  # initialize the score (for each agent)
        while True:
            actions = agent.act(states)

            env_info = env.step(actions)[brain_name]
            next_states = (
                env_info.vector_observations
            )  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished

            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards  # update the score (for each agent)
            if np.any(dones):  # exit loop if episode finished
                break
        scores_deque.append(score)
        scores.append(score)
        avg_scores.append(np.mean(scores_deque))
        if avg_scores[-1] >= goal and len(scores_deque) == window_len:
            if goal == original_goal:
                print(
                    f"\nEnvironment solved in {i_episode-100:d} episodes!"
                    f"\tAverage Score: {avg_scores[-1]:.2f}"
                )
            else:
                print(f"\nSaving better agent with Average Score: {avg_scores[-1]:.2f}")
            torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth")
            goal = int(np.mean(scores_deque)) + 1
        if i_episode % print_every == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(i_episode, avg_scores[-1])
            )

    return scores, avg_scores


# -

scores, avg_scores = ddpg()

# +
import pickle

with open("scores.pkl", "ab") as f:
    pickle.dump(scores, f)
# -

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.plot(np.arange(len(scores)), avg_scores, "r")
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.show()

env.close()
