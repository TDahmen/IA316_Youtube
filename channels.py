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
    def random_channel(shape: int = 1, nb_videos: int = 1, channel_id: int = 0):
        """ Returns a channel of random videos, all similar to a base random video
        :param shape: int or tuple of ints, specifying the shape of the keywords parameter
        :param nb_videos: int, specifying the number of videos on the channel
        :param channel_id: int, unique identifier amongst all channels
        """

        # Type-checking
        if not isinstance(shape, int):
            raise TypeError
        if not isinstance(nb_videos, int):
            raise TypeError
        if not isinstance(channel_id, int):
            raise TypeError

        videos = []

        if nb_videos > 0:

            video_id = channel_id * 10**3
            base_video = Video.random_video(shape, video_id, channel_id)
            videos.append(base_video)
            base_keywords = base_video.keywords

            for i in range(nb_videos - 1):
                switch_proba = 0.1
                switch = np.random.uniform(0, 1, size=shape)
                switch = np.asarray(switch < switch_proba, dtype=np.float)  # bernoulli sampling, p << 0.5
                keywords = np.logical_xor(base_keywords, switch)
                keywords = np.asarray(keywords, dtype=np.float)
                video = Video(keywords, video_id + i + 1, channel_id)
                videos.append(video)

        return Channel(videos, channel_id)
