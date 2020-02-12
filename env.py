import numpy as np

from users import User
from channels import Channel
from typing import List


class YoutubeEnv:

    def __init__(self, users: List[User] = [], channels: List[Channel] = [], seed: int = 0):

        # Type-checking
        if type(users) != list or not all(isinstance(x, User) for x in users):
            raise TypeError
        if type(channels) != list or not all(isinstance(x, Channel) for x in channels):
            raise TypeError
        if not isinstance(seed, int):
            raise TypeError

        self.users = users
        self.channels = channels
        self.rng = np.random.RandomState(seed)  # random number generator

        self.nb_users = len(users)
        self.keywords = np.array([u.tastes for u in users])
        self.videos = [v for c in channels for v in c.videos]
        self.videos = {v.video_id: v for v in self.videos}

        self.state = {"current_user": 0}

    def add_user(self, user):
        self.users.append(user)
        self.nb_users += 1
        self.keywords = np.vstack([self.keywords, user.tastes])

    def add_channel(self, channel):
        self.channels.append(channel)
        for v in channel.videos:
            self.videos[v.video_id] = v

    def reset(self, seed=0):
        self.state = {"current_user": 0}
        self.seed = seed

    def step(self):
        pass


u = User()
c = Channel()
print(type([u]))
env = YoutubeEnv(users=[u], channels=[c], seed=1)
print("ok")
