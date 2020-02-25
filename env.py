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

        self.users = {u.user_id: u for u in users}
        self.channels = channels
        self.rng = np.random.RandomState(seed)  # random number generator

        self.nb_users = len(users)
        self.keywords = np.array([u.keywords for u in users])
        self.videos = [v for c in channels for v in c.videos]
        self.videos = {v.video_id: v for v in self.videos}

        self.state = {"current_user": 0}

    def add_user(self, user):
        self.users[user.user_id] = user
        self.nb_users += 1
        self.keywords = np.vstack([self.keywords, user.keywords])

    def add_channel(self, channel):
        self.channels.append(channel)
        for v in channel.videos:
            self.videos[v.video_id] = v

    def reset(self, seed: int = 0):
        self.state = {"current_user": 0}
        self.rng = np.random.RandomState(seed)

    def step(self):  # TODO: define this
        pass

    @staticmethod
    def random_env(nb_keywords: int = 10, nb_users: int = 10, ku_ratio: int = 3,
                   nb_channels: int = 10, vc_ratio: int = 3, kv_ratio: int = 3, seed: int = 0):
        """ Returns a random environment.
        :param nb_keywords: total number of possible keywords
        :param nb_users: int, number of users
        :param ku_ratio: int, number of keywords per user
        :param nb_channels: int, number of channels
        :param vc_ratio: int, number of videos per channel
        :param kv_ratio: int, number of keywords per video
        :param seed: int, random seed to use
        """

        # Type-checking
        if not isinstance(nb_users, int):
            raise TypeError
        if not isinstance(ku_ratio, int):
            raise TypeError
        if not isinstance(nb_channels, int):
            raise TypeError
        if not isinstance(vc_ratio, int):
            raise TypeError
        if not isinstance(kv_ratio, int):
            raise TypeError
        if not isinstance(seed, int):
            raise TypeError

        users = []
        channels = []
        seed_0 = seed

        for user_id in range(nb_users):
            seed += 1
            users.append(User.random_user(nb_keywords, ku_ratio, user_id, seed))

        for channel_id in range(nb_channels):
            channels.append(Channel.random_channel(nb_keywords, vc_ratio, kv_ratio, channel_id))

        return YoutubeEnv(users, channels, seed_0)


#seed = 0
#env = YoutubeEnv.random_env(seed=seed)

# for video_id, video in env.videos.items():
#     print("video_id = {}, video.keywords = {}\n".format(video_id, video.keywords))

#u = env.users[0]
#c = env.channels[0]
#for v in c.videos:
#    print(u.watch(v))
