import os
import sys
import numpy as np
import torch

from pycochleagram.cochleagram import invert_cochleagram
from torchvision import transforms
from utils import waveFromCochleagram, batchWaveFromCochleagram
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy
import librosa
from librosa.feature import spectral_centroid

class Evaluator:
    def __init__(self, wavs, gt_wavs, num_batches=250):
        self.wavs = wavs
        self.gt_wavs = gt_wavs
        self.num_batches = num_batches
        
    def get_metrics(self):
        metrics = {}
        metrics["reconstructable"] = self.reconstructable()
        metrics["reconstruction_loss"] = self.reconstructable()
        metrics["loudness_diff"] = self.reconstructable()
        metrics["spectral_centroid_diff"] = self.reconstructable()
        metrics["peak_displacement"] = self.reconstructable()
    
    def _plot_metric(self, metric_arr, metric=""):
        plt.hist(metric_arr)
        plt.title(f"{metric.capitalize()} Distribution")
        plt.xlabel(metric)
        plt.show()
    
    def _compute_mse(self, y1, y2):
        return np.mean((y1 - y2)**2, axis=1)
    
    def _compute_rmse(self, y1, y2):
        '''standard difference metric for waveforms'''
        return np.sqrt(np.mean((y1 - y2)**2, axis=1))

    def _compute_cosine_similarity(self, y1, y2):
        '''Used between normalized waveforms'''
        dot = np.sum(np.multiply(y1, y2), axis=1)
        norm = np.linalg.norm(y1, axis=1) * np.linalg.norm(y2, axis=1)
        return dot/norm

    def _pearson_correlation_coefficient(self, y1, y2):
        '''Used to show correlation between rising and falling of the waveforms'''
        sample_pcc = []
        for y, _y in zip(y1, y2):
            res = scipy.stats.pearsonr(y, _y)
            sample_pcc.append(res[0])
        
        return np.array(sample_pcc)
            
    def reconstructable(self, wavs):
        not_reconstructable = 0
        for wav in self.wavs:
            if np.isnan(wav).any():
                not_reconstructable += 1
        print(not_reconstructable, len(self.wavs))
        return 1 - not_reconstructable/len(self.wavs)
    
    def reconstruction_loss(self, plot=False):
        '''Uses the RMSE metric for finding difference between 2 waveforms'''
        # Make sure to filter self.wavs and gt_wavs that cannot be reconstructed before this
        sample_rmse = self._compute_rmse(self.wavs, self.gt_wavs)

        if plot:
            self._plot_metric(sample_rmse, "reconstruction loss")

        return np.mean(sample_rmse)

    def loudness(self, plot=False):
        sample_loudness = self._compute_mse(self.wavs, self.gt_wavs)

        if plot:
            self._plot_metric(sample_loudness, "loudness loss")

        return np.mean(sample_loudness)

    def spectral_centroid_difference(self, plot=False):
        sample_centroid_difference = []
        for wav, gt_wav in zip(self.wavs, self.gt_wavs):
            # for each time_step one spectral centroid, indication of the domininant frequency that can be heard
            wav_centroids = spectral_centroid(y=wav[22560:25440]+0.01, sr=self.sr)
            gt_centroids = spectral_centroid(y=gt_wav[22560:25440]+0.01, sr=self.sr)


            '''
            plt.plot(range(wav_centroids.shape[1]), wav_centroids[0], label="pred")
            plt.plot(range(gt_centroids.shape[1]), gt_centroids[0], label="gt")
            plt.legend()
            plt.show()
            '''

            sample_centroid_difference.append(np.mean(self._compute_rmse(wav_centroids, gt_centroids)))
        
        if plot:
            self._plot_metric(sample_centroid_difference, "spectral centroid difference")

        return np.mean(sample_centroid_difference)


    def peak_displacement(self, plot=False):
        wavs_peak_pos = np.argmax(self.wavs, axis=1)
        gt_wavs_peak_pos = np.argmax(self.gt_wavs, axis=1)
        wavs_peak_pos = np.expand_dims(wavs_peak_pos, axis=-1)
        gt_wavs_peak_pos = np.expand_dims(gt_wavs_peak_pos, axis=-1)
        sample_peak_displacement = self._compute_rmse(wavs_peak_pos, gt_wavs_peak_pos)/96

        if plot:
            self._plot_metric(sample_peak_displacement, "peak displacement difference")
        
        return np.mean(sample_peak_displacement)

    def sample_inference_time(self, wavs, plot=False):
        pass

    def material_consistency(self, wavs, gt_wavs, plot=False):
        pass