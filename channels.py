from videos import Video
from typing import List


class Channel:

    def __init__(self, videos: List[Video] = [], channel_id: int = 0):

        # Type-checking
        if type(videos) != list or not all(isinstance(x, Video) for x in videos):
            raise TypeError
        if not isinstance(channel_id, int):
            raise TypeError

        self.videos = videos
        self.channel_id = channel_id  # supposed to be a unique identifier amongst all channels



