import os
import sys
import numpy as np

from pycochleagram.cochleagram import invert_cochleagram

class NoStdStreams(object):
    """
    Utility class to silence all output from the code within a block. Usage:

    with NoStdStreams():
        *code here*
    """
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()

def waveFromCochleagram(cochleagram:np.ndarray):

    with NoStdStreams():
        wave = invert_cochleagram(cochleagram, 96000, 40, 100, 10000, 1, downsample=90, nonlinearity='power')[0]
    return wave