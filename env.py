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

    def reset(self, seed: int = 0):
        self.state = {"current_user": 0}
        self.rng = np.random.RandomState(seed)

    def step(self):
        pass

    @staticmethod
    def random_env(nb_tastes: int = 100, nb_users: int = 10, tu_ratio: int = 3, nb_channels: int = 10, vc_ratio: int = 3, seed: int = 0):
        """ Returns a random environment.
        :param nb_tastes: total number of possible tastes
        :param nb_users: int, number of users
        :param tu_ratio: int, number of tastes per user
        :param nb_channels: int, number of channels
        :param vc_ratio: int, number of videos per channel
        :param seed: int, random seed to use
        """

        # Type-checking
        if not isinstance(nb_users, int):
            raise TypeError
        if not isinstance(tu_ratio, int):
            raise TypeError
        if not isinstance(nb_channels, int):
            raise TypeError
        if not isinstance(vc_ratio, int):
            raise TypeError
        if not isinstance(seed, int):
            raise TypeError

        users = []
        channels = []
        seed_0 = seed

        for _ in range(nb_users):
            seed += 1
            users.append(User.random_user(nb_tastes, tu_ratio, seed))

        for channel_id in range(nb_channels):
            channels.append(Channel.random_channel(nb_tastes, vc_ratio, channel_id))

        return YoutubeEnv(users, channels, seed_0)


seed = 0
env = YoutubeEnv.random_env(seed=seed)

# for video_id, video in env.videos.items():
#     print("video_id = {}, video.keywords = {}\n".format(video_id, video.keywords))

u = env.users[0]
c = env.channels[0]
for v in c.videos:
    print(u.watch(v))
