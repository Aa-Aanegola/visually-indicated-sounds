import cv2
import numpy as np

from typing import List, Tuple

class VISDataPoint:

    """
    A single data point is a sonic and visual representation of a 0.5 second window around a sonic peak.
    The sonic component is a cochleagram of the shape 42x45, where 45 is along the temporal axis (0.5 seconds with a 90Hz sampling rate) and 42 is along the frequency axis (42 bands)
    The visual component is a list of spacetime 15 images (0.5 seconds with 30FPS frame rate), where each spacetime image is an image with 3 channels, containing the grayscale images at times t-1, t and t+1

    For the sake of storage efficiency, we'll only store the 15 grayscale frames, and build the spacetime images on retreival time.
    We also duplicate each spacetime image 3 times on retreival time so that the cochleagram and the spacetime images list are of the same length 
    """

    def __init__(self, cochleagram:np.ndarray, frames:List[np.ndarray], material:str) -> None:

        self._cochleagram = cochleagram.astype(np.float16)
        self._frame0 = frames[1]
        self._frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.uint8) for f in frames]
        self._material = material

    @property
    def cochleagram(self) -> np.ndarray:
        return self._cochleagram

    @property
    def frames(self) -> Tuple[List[np.ndarray], np.ndarray]:
        spaceTimeFrames = []
        for i in range(1, len(self._frames)-1):
            spaceTimeFrame = np.dstack([self._frames[i-1], self._frames[i], self._frames[i+1]])
            spaceTimeFrames += [spaceTimeFrame] * 3
        return spaceTimeFrames, self._frame0

    @property
    def material(self) -> str:
        return self._material