import cv2
import numpy as np
import os
import pandas as pd

import tickle

from pycochleagram.utils import wav_to_array
from pycochleagram.cochleagram import human_cochleagram
from tqdm import tqdm

from VISDataPoint import VISDataPointV2
from utils import NoStdStreams

from joblib import Parallel, delayed
from multiprocessing import Lock, Manager

import warnings
warnings.filterwarnings("ignore")

def createDatapointsFromFile(file_name, frame_size=(224,224), window_duration=0.5):
    wav_file = os.path.join(root, f'{file_name}_denoised.wav')
    video_file = os.path.join(root, f'{file_name}_denoised.mp4')
    annotation_file = os.path.join(root, f'{file_name}_times.txt')

    annotations = pd.read_csv(annotation_file, sep=' ', names=['Time', 'Material', 'Contact Type', 'Motion Type'])
    wav, sample_rate = wav_to_array(wav_file)

    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        resized_frame = cv2.resize(frame, dsize=frame_size, interpolation=cv2.INTER_CUBIC)
        frames.append(resized_frame)
    
    cap.release()

    data_points = []
    for row in annotations.iterrows():
        peak_time = row[1]['Time']
        start_time = peak_time - window_duration/2

        start_frame = int(start_time * frame_rate)
        end_frame = start_frame + int(frame_rate * window_duration)
        window_frames = frames[start_frame:end_frame+2]

        start_sound = int(start_time * sample_rate)
        end_sound = start_sound + int(sample_rate * window_duration)
        window_sound = wav[start_sound:end_sound]

        coch = human_cochleagram(window_sound, sample_rate, n=40, low_lim=100, hi_lim=10000, sample_factor=1, downsample=90, nonlinearity='power')

        data_points.append(VISDataPointV2(coch, window_frames, row[1]['Material']))
    
    return data_points

def processFile(file_name, idx, total_files):
    print(f"Processing file {idx+1}/{total_files}", end='\r')
    global readLock
    try:
        with NoStdStreams():
            data_points = createDatapointsFromFile(file_name)
        for data_point in data_points:
            readLock.acquire()
            n_points = sharedMem['n_points']
            sharedMem['n_points'] += 1
            readLock.release()
            tickle.dump(data_point, f'/scratch/kapur/test/{n_points}.tkl')
    except:
        pass

manager = Manager()
sharedMem = manager.dict()
readLock = manager.RLock()

if __name__ == "__main__":
    with open('../../data/test.txt', 'r') as f:
        file_names = [x.strip() for x in f.readlines()]

    root = '../../data/'

    sharedMem['n_points'] = 0

    Parallel(n_jobs=-1)(delayed(processFile)(file_name, idx, len(file_names)) for (idx, file_name) in enumerate(file_names))

    print()
    print(f"Total Data Points written: {sharedMem['n_points']}")