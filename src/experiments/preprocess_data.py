import pandas as pd
from pycochleagram.utils import wav_to_array
from pycochleagram.cochleagram import human_cochleagram
import cv2
import numpy as np
import scipy
import torchvision.transforms as T
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle
import sys
import os
sys.stdout = open(os.devnull, 'w')

class VISDataset(Dataset):
    def __init__(self, root, coch_root, dataset_file, window_duration=0.5, datum_len=45, transform=T.Compose([T.Resize((224, 224), antialias=False)]),is_eval=False):
        self.root = root
        self.coch_root = coch_root
        
        with open(f'{self.root}/{dataset_file}', 'r') as f:
            self.file_list = [file.strip() for file in f.readlines()]
        
        self.is_eval = is_eval
        self.transform = transform
        
        self.video_fps = 30
        self.window_duration = window_duration
        self.n_frames = int(window_duration * self.video_fps)
        assert datum_len % self.n_frames == 0
        self.n_tiles = datum_len // self.n_frames
        
        self.data = []
        
        for file in tqdm(self.file_list):
            try:    
                vid = cv2.VideoCapture(f'{self.root}/{file}_denoised.mp4')
                frames = []
                while True:
                    ret, frame = vid.read()
                    if not ret:
                        break
                    frames.append(frame)
                vid.release()
                
                wav, sample_rate = wav_to_array(f'{self.root}/{file}_denoised.wav')
                annotations = pd.read_csv(f'{self.root}/{file}_times.txt', sep=' ', names=['Time', 'Material', 'Action', 'Reaction'])
                
                cochleagrams = scipy.io.loadmat(f'{self.coch_root}/{file}_sf.mat')['sfs']
                
                for ind, row in annotations.iterrows():
                    datum = {}
                    peak_time = row['Time']
                    peak_vid = int(peak_time * self.video_fps)
                    frames_rgb = np.stack(frames[peak_vid-self.n_frames//2:1+peak_vid+self.n_frames//2]).transpose(0, 3, 1, 2)
                    frames_spacetime = np.stack([self.get_spacetime(frames[i-1:i+2]) for i in range(peak_vid-self.n_frames//2, 1+peak_vid+self.n_frames//2)])
                    # frames_rgb = np.repeat(frames_rgb, self.n_tiles, axis=0).transpose(0, 3, 1, 2)
                    # frames_spacetime = np.repeat(frames_spacetime, self.n_tiles, axis=0)                    
                    frames_rgb = self.transform(torch.tensor(frames_rgb))
                    frames_spacetime = self.transform(torch.tensor(frames_spacetime))
                    
                    start_time = peak_time - window_duration/2
                    end_time = peak_time + window_duration/2
                    start_frame = int(start_time * sample_rate)
                    end_frame = int(end_time * sample_rate)
                    peak = wav[start_frame:end_frame]
                    coch = human_cochleagram(peak, sample_rate, n=40, low_lim=100, hi_lim=10000, sample_factor=1, downsample=90, nonlinearity='power')
                            
                    datum['frames_rgb'] = frames_rgb
                    datum['frames_spacetime'] = frames_spacetime
                    # datum['og_cochleagram'] = torch.tensor(cochleagrams[ind])
                    datum['cochleagram'] = torch.tensor(coch, dtype=torch.float16).transpose(1, 0)
                    datum['material'] = row['Material']
                    # datum['action'] = row['Action']
                    # datum['reaction'] = row['Reaction']
                    # print(datum['frames_rgb'].shape, datum['frames_spacetime'].shape, datum['frames_rgb'].dtype, datum['frames_spacetime'].dtype, datum['cochleagram'].shape, datum['cochleagram'].dtype, datum['action'], datum['material'], datum['reaction'])
                    # self.data.append(datum)
                    
                    with open(f'../../data/preprocessed/{file}-{ind}.pkl', 'wb') as f:
                        pickle.dump(datum, f)                    
                    
            except:
                pass
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)           
    
    def get_spacetime(self, frames):
        return np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames])

    # def dump(self, root):
    #     for ind, datum in enumerate(self.data):
    #         with open(f'{root}/{ind}.pkl', 'wb') as f:
    #             pickle.dump(datum, f)
                

if __name__ == '__main__':
    ds = VISDataset('../../data/vis-data-256', '../../data/vis-data', 'train.txt')
    # ds.dump('../../data/preprocessed')