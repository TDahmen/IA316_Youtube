import numpy as np


class Video:

    def __init__(self, keywords: np.ndarray = np.array([], dtype=np.float), video_id: int = 0, channel_id: int = 0):

        # Type-checking
        if not isinstance(keywords, np.ndarray):
            raise TypeError
        if keywords.dtype != np.float:
            raise TypeError
        if not isinstance(video_id, int):
            raise TypeError
        if not isinstance(channel_id, int):
            raise TypeError

        self.keywords = keywords
        self.video_id = video_id  # supposed to be a unique identifier amongst all videos
        self.channel_id = channel_id  # supposed to be a unique identifier amongst all channels

    @staticmethod
    def random_video(shape: int = 1, video_id: int = 0, channel_id: int = 0):
        """ Returns a video with random keywords
        :param shape: int or tuple of ints, specifying the shape of the keywords parameter
        :param video_id: int, unique identifier amongst all videos
        :param channel_id: int, unique identifier amongst all channels
        """

        # Type-checking
        if not isinstance(shape, int):
            raise TypeError

        keywords = np.random.uniform(0, 1, size=shape)  # random tastes
        keywords = np.asarray(keywords < 0.5, dtype=np.float)  # bernoulli sampling, p = 0.5
        return Video(keywords, video_id, channel_id)



