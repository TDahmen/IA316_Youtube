from typing import List

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



