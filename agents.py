import numpy as np
import random
import matplotlib.pyplot as plt

from users import User
from channels import Channel, Video
from typing import List
from env import YoutubeEnv


class Agent:

    def __init__(self, user: User, env: YoutubeEnv):
        """
        Agent for Youtube Environment.
        :param user: User object
        :param env: YoutubeEnv object
        """

        self.user = user
        self.env = env

        self.actions = list(env.videos.values())

    def thompson(self, nb_tries, cum_rewards, param = None):
        k = np.shape(nb_tries)[0]
        if param == "beta":
            # Beta prior
            try:
                samples = np.random.beta(cum_rewards + 1, nb_tries - cum_rewards + 1)
            except:
                samples = np.random.random(k)
        else:
            # Normal prior
            samples = np.random.normal(cum_rewards / (nb_tries + 1), 1. / (nb_tries + 1))
        return np.argmax(samples)

    def thompson_sim(self, time_horizon, prior = None):
        k = len(self.actions)
        nb_tries = np.zeros(k, int)
        cum_rewards = np.zeros(k, float)
        action_seq = []
        reward_seq = []
        for t in range(time_horizon):
            a = self.thompson(nb_tries, cum_rewards, prior)
            r = self.user.watch(self.actions[a])
            if self.env.evolutive:
                env.update(self.user, self.actions[a], r)
            nb_tries[a] += 1
            cum_rewards[a] += r
            action_seq.append(a)
            reward_seq.append(r)
        index = np.where(nb_tries > 0)[0]
        best_action = index[np.argmax(cum_rewards[index] / nb_tries[index])]
        return action_seq, reward_seq

    def qlearning_sim(self, time_horizon, alpha = 0.7, gamma = 0.3, epsilon = 0.5):
    
        q_table = np.zeros(len(self.actions))

        action_seq = []
        reward_seq = []

        for i in range(0, time_horizon):

            epochs, penalties, reward, = 0, 0, 0
            
            if random.uniform(0, 1) < epsilon:
                action = random.sample([i for i in range(0, len(self.actions))], 1)[0] # Explore action space
            else:
                action = np.argmax(q_table) # Exploit learned values
            print(action)
            reward = self.user.watch(self.actions[action]) 
            
            old_value = q_table[action]
            next_max = np.max(q_table)
            
            new_value =  (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[action] = new_value
            
            action_seq.append(action)
            reward_seq.append(reward)

            if reward == 0:
                penalties += 1

            epochs += 1

        return action_seq, reward_seq

    def get_best_action(self):
        best_action, best_reward = (0, 0)
        for i in range(len(self.actions)):
            rew = self.user.watch(self.actions[i])
            if rew > best_reward:
                best_action = i
                best_reward = rew
        return best_action, best_reward



env = YoutubeEnv.random_env(seed=42)
user = env.users[0]

agent = Agent(user, env)

thompsonRes = agent.thompson_sim(300)
qlearningRes = agent.qlearning_sim(300)

bestAction = agent.get_best_action()

## Fonctions outil

def get_regret(action_seq, reward_seq, best_actions, best_reward):
    time_horizon = len(action_seq)
    regret = np.zeros(time_horizon, float)
    precision = np.zeros(time_horizon, float)
    for t in range(time_horizon):
        regret[t] = best_reward - reward_seq[t]
    return np.cumsum(regret), precision

def show_metrics(metrics, time_horizon):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 4))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Regret')
    ax1.plot(range(time_horizon),metrics[0], color = 'b')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Precision')
    ax2.set_ylim(-0.02,1.02)
    ax2.plot(range(time_horizon),metrics[1], color = 'b')
    plt.show()

##

print("cumRegret Thompson : ")

regretThompson = get_regret(*thompsonRes, *bestAction)
show_metrics(regretThompson, 300)

print("cumRegret qlearning : ")

regretQlearning = get_regret(*qlearningRes, *bestAction)
show_metrics(regretQlearning, 300)











