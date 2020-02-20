import numpy as np
from scipy.spatial.distance import cosine

from videos import Video
from channels import Channel


cosine_sim = lambda u, v: 1 - cosine(u, v)  # redefine cosine distance as cosine similarity

class User:

    def __init__(self, tastes: np.ndarray = np.array([], dtype=np.float), seed: int = 0):

        # Type-checking
        if not isinstance(tastes, np.ndarray):
            raise TypeError
        if tastes.dtype != np.float:
            raise TypeError
        if not isinstance(seed, int):
            raise TypeError

        # Normalization check
        if not all(x >= 0 and x <= 1 for x in tastes):
            raise AssertionError(
                "tastes argument given for initialization of User object should have entries in [0 ,1]")

        self.tastes = tastes
        self.videos_liked = set()
        self.videos_disliked = set()
        self.to_watch_later_playlist = set()
        self.channels_suscribed = set()
        self.rng = np.random.RandomState(seed)  # random number generator

    def like(self, video: Video):
        self.videos_liked.add(video.video_id)

    def dislike(self, video: Video):
        self.videos_disliked.add(video.video_id)

    def watch_later(self, video: Video):
        self.to_watch_later_playlist.add(video.video_id)

    def suscribe(self, channel: Channel):
        self.channels_suscribed.add(channel.channel_id)

    def watch(self, video: Video):
        """ Returns a random watch time of a video based on personal tastes and video keywords """
        sim = cosine_sim(self.tastes, video.keywords)  # cosine similarity, mean of the random draw
        # var = (sim * (1 - sim)) / 2  # variance of the random draw
        alpha = sim
        beta = 1 - sim
        watch_time = self.rng.beta(alpha, beta)
        return watch_time


