import numpy as np
import random
import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine

from users import User
from channels import Channel, Video
from typing import List
from env import YoutubeEnv


def cosine_sim(u, v):
    """ Redefine cosine distance as cosine similarity, with numerical stability """
    sim = 1 - cosine(u, v)
    epsilon = 10e-8
    sim = max(epsilon, sim)
    sim = min(1 - epsilon, sim)
    return sim

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

    def eps_greedy(self, nb_tries, cum_rewards, param=None):
        if param == None:
            eps = 0.1
        else:
            eps = float(param)
        k = np.shape(nb_tries)[0]
        if np.sum(nb_tries) == 0 or np.random.random() < eps:
            return np.random.randint(k)
        else:
            index = np.where(nb_tries > 0)[0]
            return index[np.argmax(cum_rewards[index] / nb_tries[index])]

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
                self.env.update(self.user, self.actions[a], r)
            nb_tries[a] += 1
            cum_rewards[a] += r
            action_seq.append(a)
            reward_seq.append(r)
        index = np.where(nb_tries > 0)[0]
        best_action = index[np.argmax(cum_rewards[index] / nb_tries[index])]
        return action_seq, reward_seq

    def eps_greedy_sim(self, time_horizon, prior = None):
        k = len(self.actions)
        nb_tries = np.zeros(k, int)
        cum_rewards = np.zeros(k, float)
        action_seq = []
        reward_seq = []
        for t in range(time_horizon):
            a = self.eps_greedy(nb_tries, cum_rewards, prior)
            r = self.user.watch(self.actions[a])
            if self.env.evolutive:
                self.env.update(self.user, self.actions[a], r)
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
            reward = self.user.watch(self.actions[action])
            if self.env.evolutive:
                self.env.update(self.user, self.actions[action], reward)
            
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


    def get_best_action_sim(self):
        best_action, best_sim = (0, 0)
        for i in range(len(self.actions)):
            sim = cosine_sim(self.user.keywords, self.actions[i].keywords)
            if sim > best_sim:
                best_action = i
                best_sim = sim
        return best_action, best_sim

## Fonctions outil

def get_regret(reward_seq, best_actions, best_reward):
    time_horizon = len(reward_seq)
    regret = np.zeros(time_horizon, float)
    precision = np.zeros(time_horizon, float)
    for t in range(time_horizon):
        regret[t] = best_reward - reward_seq[t]
    return np.cumsum(regret), precision

def show_metrics(metrics, titles, time_horizon):

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 4))

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Regret')
    ax1.title.set_text(titles[0])
    ax1.plot(range(time_horizon),metrics[0], color = 'b')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Precision')
    ax2.set_ylim(-0.02,1.02)
    ax2.title.set_text(titles[1])
    ax2.plot(range(time_horizon),metrics[1], color = 'b')
    plt.show()











