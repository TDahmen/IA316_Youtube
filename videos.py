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

        # Normalization
        keywords = keywords / np.sum(keywords)

        self.keywords = keywords
        self.video_id = video_id  # supposed to be a unique identifier amongst all videos
        self.channel_id = channel_id  # supposed to be a unique identifier amongst all channels

    @staticmethod
    def random_video(nb_keywords: int = 1, kv_ratio: int = 1, video_id: int = 0, channel_id: int = 0):
        """ Returns a video with random keywords
        :param nb_keywords: int or tuple of ints, specifying the length of the keywords parameter
        :param kv_ratio: int, specifying a number of non-zeros keywords for the video
        :param video_id: int, unique identifier amongst all videos
        :param channel_id: int, unique identifier amongst all channels
        """

        # Type-checking
        if not isinstance(nb_keywords, int):
            raise TypeError
        if not isinstance(kv_ratio, int):
            raise TypeError
        # Assertion checking
        assert kv_ratio <= nb_keywords

        keywords_idx = np.random.choice(np.arange(nb_keywords), size=kv_ratio, replace=False)
        keywords = np.zeros(nb_keywords, dtype=np.float)
        keywords[keywords_idx] = 1.  # non-zeros keywords for the video
        return Video(keywords, video_id, channel_id)
