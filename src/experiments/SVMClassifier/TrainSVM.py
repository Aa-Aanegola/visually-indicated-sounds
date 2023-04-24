import sys
sys.path.append('../../')

import numpy as np
import tickle
import glob
import pickle

from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.svm import SVC
from VISDataPoint import VISDataPointV3

materials = ['None',
            'rock',
            'leaf',
            'water',
            'wood',
            'plastic-bag',
            'ceramic',
            'metal',
            'dirt',
            'cloth',
            'plastic',
            'tile',
            'gravel',
            'paper',
            'drywall',
            'glass',
            'grass',
            'carpet']

def get_data(file_name):
    dataPoint: VISDataPointV3 = tickle.load(file_name)
    return dataPoint.wav, materials.index(dataPoint.material)

select_ratio = 0.4

train_files = glob.glob('/scratch/vis_data_v3/train/*.tkl')
train_files = [f for f in train_files if np.random.random() < select_ratio]

train_data = Parallel(n_jobs=8)(delayed(get_data)(file_name) for file_name in tqdm(train_files))

X_train = [x[0] for x in train_data]
y_train = [x[1] for x in train_data]

classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)

with open('svm.pkl', 'wb') as f:
    pickle.dump(classifier, f)