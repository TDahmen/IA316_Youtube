from typing import List


class Video:

    def __init__(self, keywords: List[int] = [], video_id: int = 0):

        # Type-checking
        if (not isinstance(keywords, list)) or (not all(isinstance(x, int) for x in keywords)):
            raise TypeError
        if not isinstance(video_id, int):
            raise TypeError

        self.keywords = keywords
        self.video_id = video_id  # supposed to be a unique identifier amongst all videos


