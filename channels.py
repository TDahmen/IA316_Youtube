from videos import Video
from typing import List

import numpy as np


class Channel:

    def __init__(self, videos: List[Video] = [], channel_id: int = 0):

        # Type-checking
        if type(videos) != list or not all(isinstance(x, Video) for x in videos):
            raise TypeError
        if not isinstance(channel_id, int):
            raise TypeError

        self.videos = videos
        self.channel_id = channel_id  # supposed to be a unique identifier amongst all channels

    @staticmethod
    def random_channel(nb_keywords: int = 1, nb_videos: int = 1, kv_ratio: int = 1, channel_id: int = 0):
        """ Returns a channel of random videos, all similar to a base random video
        :param nb_keywords: int or tuple of ints, specifying the length of the keywords parameter
        :param nb_videos: int, specifying the number of videos on the channel
        :param kv_ratio: int, number of keywords per video
        :param channel_id: int, unique identifier amongst all channels
        """

        # Type-checking
        if not isinstance(nb_keywords, int):
            raise TypeError
        if not isinstance(nb_videos, int):
            raise TypeError
        if not isinstance(kv_ratio, int):
            raise TypeError
        if not isinstance(channel_id, int):
            raise TypeError

        videos = []

        if nb_videos > 0:

            video_id = channel_id * 10**3
            base_video = Video.random_video(nb_keywords, kv_ratio, video_id, channel_id)
            videos.append(base_video)
            base_keywords = base_video.keywords

            for i in range(nb_videos - 1):
                switch_proba = 0.1
                switch = np.random.uniform(0, 1, size=nb_keywords)
                switch = np.asarray(switch < switch_proba, dtype=np.float)  # bernoulli sampling, p << 0.5
                keywords = np.logical_xor(base_keywords, switch)
                keywords = np.asarray(keywords, dtype=np.float)
                video = Video(keywords, video_id + i + 1, channel_id)
                videos.append(video)

        return Channel(videos, channel_id)
