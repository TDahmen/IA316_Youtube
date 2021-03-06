import numpy as np
from scipy.spatial.distance import cosine

from videos import Video
# from channels import Channel


def cosine_sim(u, v):
    """ Redefine cosine distance as cosine similarity, with numerical stability """
    sim = 1 - cosine(u, v)
    epsilon = 10e-8
    sim = max(epsilon, sim)
    sim = min(1 - epsilon, sim)
    return sim


class User:

    def __init__(self, keywords: np.ndarray = np.array([], dtype=np.float), user_id: int = 0, seed: int = 0):

        # Type-checking
        if not isinstance(keywords, np.ndarray):
            raise TypeError
        if keywords.dtype != np.float:
            raise TypeError
        if not isinstance(user_id, int):
            raise TypeError
        if not isinstance(seed, int):
            raise TypeError

        # Normalization
        keywords = keywords / np.sum(keywords)

        self.keywords = keywords
        self.original_keywords = keywords
        self.user_id = user_id
        self.history = list()  # list of tuples (video_id, watch_time) of videos watched in the past
        # self.videos_liked = set()
        # self.videos_disliked = set()
        # self.to_watch_later_playlist = set()
        # self.channels_suscribed = set()
        self.rng = np.random.RandomState(seed)  # random number generator
        self.original_seed = seed

    def add_to_history(self, video_id: int = 0, watch_time: float = 0.):

        # type-checking
        if not isinstance(video_id, int):
            raise TypeError
        if not isinstance(watch_time, float):
            raise TypeError

        self.history.append((video_id, watch_time))

    def evolve(self, video: Video, watch_time: float = 0., gamma: float = 0.05):
        """
        Models the evolution of taste
        :param video: Video, video watched by the user
        :param watch_time: effective watch time of the video by the user
        :param gamma: evolution parameter (learning rate)
        :return:
        """

        # type-checking
        if not isinstance(video, Video):
            raise TypeError
        if not isinstance(watch_time, float):
            raise TypeError
        if not isinstance(gamma, float):
            raise TypeError

        if self.rng.uniform() < watch_time:  # bernoulli sampling with p = watch_time
            # fluctuate tastes around original tastes, current tastes and video content
            keywords = self.original_keywords + self.keywords + gamma * video.keywords
            keywords = keywords / np.sum(keywords)
            self.keywords = keywords  # evolution of taste

    # def like(self, video: Video):
    #     self.videos_liked.add(video.video_id)

    # def dislike(self, video: Video):
    #     self.videos_disliked.add(video.video_id)

    # def watch_later(self, video: Video):
    #     self.to_watch_later_playlist.add(video.video_id)

    # def suscribe(self, channel: Channel):
    #     self.channels_suscribed.add(channel.channel_id)

    def watch(self, video: Video):
        """ Returns a random watch time of a video based on personal keywords and video keywords """

        # Type-checking
        if not isinstance(video, Video):
            raise TypeError

        sim = cosine_sim(self.keywords, video.keywords)  # cosine similarity, mean of the random draw

        # var = (sim * (1 - sim)) / 2  # variance of the random draw
        # alpha = sim
        # beta = 1 - sim

        var = (sim * (1 - sim)) / 16
        nu = (sim * (1 - sim)) / var - 1
        alpha = sim * nu
        beta = (1 - sim) * nu

        watch_time = self.rng.beta(alpha, beta)
        return watch_time

    @staticmethod
    def random_user(nb_keywords: int = 1, ku_ratio: int = 1, user_id: int = 0, seed: int = 0):
        """ Returns a user with random keywords
        :param nb_keywords: int or tuple of ints, specifying the total number of keywords possible
        :param ku_ratio: int, specifying a number of non-zeros keywords for the user
        :param user_id: int, unique identifier amongst all users
        :param seed: int, random seed to use
        """

        # Type-checking
        if not isinstance(nb_keywords, int):
            raise TypeError
        if not isinstance(ku_ratio, int):
            raise TypeError
        # Assertion checking
        assert ku_ratio <= nb_keywords

        keywords_idx = np.random.choice(np.arange(nb_keywords), size=ku_ratio, replace=False)
        keywords = np.zeros(nb_keywords, dtype=np.float)
        keywords[keywords_idx] = np.random.uniform(0.8, 1, size=ku_ratio)  # random keywords between 0.8 and 1
        return User(keywords, user_id, seed)
