"""
Space Optimized Data Serialization Abstraction
"""

import pickle
import lzma

def dump(data, path):
    with lzma.open(path, 'wb') as f:
        pickle.dump(data, f)

def load(path):
    with lzma.open(path, 'rb') as f:
        return pickle.load(f)