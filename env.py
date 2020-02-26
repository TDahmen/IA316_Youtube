import numpy as np

from users import User
from channels import Channel, Video
from typing import List


class YoutubeEnv:

    def __init__(self, users: List[User] = [], channels: List[Channel] = [], evolutive: bool = False, seed: int = 0):
        """
        Youtube-like environment.
        :param users: list of User objects
        :param channels: list of Channel objects
        :param evolutive: boolean, if True, tastes or the users evolve over time
        :param seed: seed used for random number generation
        """

        # Type-checking
        if type(users) != list or not all(isinstance(x, User) for x in users):
            raise TypeError
        if type(channels) != list or not all(isinstance(x, Channel) for x in channels):
            raise TypeError
        if not isinstance(evolutive, bool):
            raise TypeError
        if not isinstance(seed, int):
            raise TypeError

        self.users = {u.user_id: u for u in users}
        self.channels = channels
        self.evolutive = evolutive
        self.rng = np.random.RandomState(seed)  # random number generator

        self.nb_users = len(users)
        self.keywords = np.array([u.keywords for u in users])
        self.videos = [v for c in channels for v in c.videos]
        self.videos = {v.video_id: v for v in self.videos}

        self.state = {"current_user": 0}

    def add_user(self, user: User):
        self.users[user.user_id] = user
        self.nb_users += 1
        self.keywords = np.vstack([self.keywords, user.keywords])

    def add_channel(self, channel: Channel):
        self.channels.append(channel)
        for v in channel.videos:
            self.videos[v.video_id] = v

    def reset(self, seed: int = 0):
        self.state = {"current_user": 0}
        self.rng = np.random.RandomState(seed)

    def step(self):
        """ Chooses a user at random and returns it """
        return np.random.choice(list(self.users.values()))

    def update(self, user: User, video: Video, watch_time: float):
        """ Updates the environment
        :param user: user
        :param video: video recommended to the user
        :param watch_time: effective watch time of the video by the user
        """

        # Type-checking
        if not isinstance(user, User):
            raise TypeError
        if not isinstance(video, Video):
            raise TypeError
        if not isinstance(watch_time, float):
            raise TypeError

        user.add_to_history(video.video_id, watch_time)  # history update
        if self.evolutive:  # if the environment is evolutive,
            user.evolve(video, watch_time)  # the user tastes can evolve

    @staticmethod
    def random_env(nb_keywords: int = 10, nb_users: int = 10, ku_ratio: int = 3,
                   nb_channels: int = 10, vc_ratio: int = 3, kv_ratio: int = 3,
                   evolutive: bool = False, seed: int = 0):
        """ Returns a random environment.
        :param nb_keywords: total number of possible keywords
        :param nb_users: int, number of users
        :param ku_ratio: int, number of keywords per user
        :param nb_channels: int, number of channels
        :param vc_ratio: int, number of videos per channel
        :param kv_ratio: int, number of keywords per video
        :param evolutive: boolean, if True, tastes or the users evolve over time
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
        if not isinstance(evolutive, bool):
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

        return YoutubeEnv(users, channels, evolutive, seed_0)
