# BiLSTM Model

This directory holds the modified implementation of the model architecture described by the original paper, by replacing standard LSTMs with Bidirectional LSTMs.

### Model Architecture

```mermaid
flowchart LR;

in(Spacetime Frames);
featureExtractor["Feature Extractor (ResNet18)"];
lstm[BiLSTM];
mlp[MLP Head];
out(Cochleagram);

in -->|Nx45x3x244x244|featureExtractor;
featureExtractor -->|Nx45x512|lstm;
lstm -->lstm;
lstm -->|Nx45x512|mlp;
mlp-->|Nx45x42|out;
```