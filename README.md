# Visually Indicated Sounds

Implementation and extension of the [paper](https://arxiv.org/abs/1512.08512), Visually Indicated Sounds by Andrew et al. which proposes the task of predicting what sound an object makes when struck as a way of studying physical interactions within a visual scene.

## Brief Description of the paper
The authors present an algorithm that synthesizes sound from silent videos of people hitting and scratching objects with a drumstick. This algorithm uses a recurrent neural network to predict sound features from videos and then produces a waveform from these features with an example-based synthesis procedure. The authors show that the sounds predicted by their model are realistic enough to fool participants in a “real or fake” psychophysical experiment and that they convey significant information about material properties and physical interactions.


## Implementations

All implementation code can be found inside `/src/experiments`, where each experiment is contained within its own directory.

* [`PaperModel`](src/experiments/PaperModel): The model architecture used in the paper.
* [`BiLSTMModel`](src/experiments/BiLSTMModel/): A modification to the architecture described in the paper, by replacing the LSTM with a Bidirectional LSTM.
* [`VMAEModel`:](src/experiments/VMAEModel/) Using modern transformer based architecture for Feature Extraction.
* [`LatentVMAEModel`](src/experiments/LatentVMAEModel/): Switching out Cochleagrams with a Learned Latent Space Representation of the waves through an AutoEncoder, fed into the `VMAEModel`.
