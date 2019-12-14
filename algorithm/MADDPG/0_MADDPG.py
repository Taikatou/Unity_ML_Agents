import datetime

from mlagents.envs import UnityEnvironment

from collections import deque

from MADDPG.Agents.MADDPGAgent import Agent as MA

import numpy as np

#######################################################################################
game = "Pong"
env_name = "../../env/" + game + "/Windows/" + game

run_episode = 10000
test_episode = 100

train_mode = True

scores = []

####################################################################################

env = UnityEnvironment(file_name=env_name, no_graphics=True)

####################################################################################

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

####################################################################################

# reset the environment
env_info = env.reset(train_mode=True)

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size[0]
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agent = MA(state_size, action_size, num_agents, fc1=400, fc2=300, seed=0, update_times=10)


##############################################################################


def get_action(brain_name):
    # Get next status, reward, and end of game information for first agent
    next_state = env_info[brain_name].vector_observations[0]
    reward = env_info[brain_name].rewards[0]
    done = env_info[brain_name].local_done[0]

    return next_state, reward, done


def solve_environment(n_episodes=6000):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    global scores
    global train_mode
    global env_info

    # 유니티 브레인 설정
    brain_name1 = env.brain_names[0]
    brain_name2 = env.brain_names[1]

    brain1 = env.brains[brain_name1]
    brain2 = env.brains[brain_name2]

    step = 0

    for i_episode in range(run_episode + test_episode):
        if i_episode == run_episode:
            train_mode = False

        # reset environment set learning mode
        env_info = env.reset(train_mode=train_mode)

        done = False

        agent.reset_random()  # reset noise object

        # initialize state and rewards
        state1 = env_info[brain_name1].vector_observations[0]
        episode_rewards1 = 0

        state2 = env_info[brain_name2].vector_observations[0]
        episode_rewards2 = 0

        reward_this_episode_1 = 0
        reward_this_episode_2 = 0
        while not done:
            step += 1

            next_state1, reward1, done1 = get_action(brain1)
            next_state2, reward2, done2 = get_action(brain2)

            action1 = agent.act(state1)
            action2 = agent.act(state2)

            env_info = env.step(vector_action={brain_name1: [action1], brain2: [action2]})

            # update episode rewards
            episode_rewards1 += reward1
            episode_rewards2 += reward2

            state1 = next_state1
            state2 = next_state2

            reward_this_episode_1 += reward1
            reward_this_episode_2 += reward2

            done = done1 or done2

        score = max(reward_this_episode_1, reward_this_episode_2)
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 2:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            import torch
            torch.save(agent.critic_local.state_dict(), 'trained_weights/checkpoint_critic.pth')
            torch.save(agent.actor_local.state_dict(), 'trained_weights/checkpoint_actor.pth')
            break
    return


solve_environment()
