# VideoMAE Model

This directory holds the implementation of `VISVMAEModel`, an entirely new approach to the model architecture for this problem. We replace the ResNet18 feature extractor + LSTM layer from the original implemenation with a pretrained Video Masked AutoEncoder. This takes as input a stack of 16 RGB frames and returns a single 768 dimension embedding. This embedding is fed into an MLP head who's output is an 1840 vector, which is then reshaped into a 45x42 tensor and compared against the corresponding cochleagram.

```mermaid
flowchart LR

in1(Wave)
encoder[Encoder]
decoder[Decoder]
out1("Wave (Reconstructed)")

in2(16 RGB Frames)
featureExtractor[VideoMAE]
mlp[MLP Head]

loss{Loss}

in2 -->|Nx16x3x224x224| featureExtractor
featureExtractor -->|Nx768| mlp
mlp --> |Nx2048|decoder
decoder -->|Nx48000| out1

in1 -->|Nx48000| encoder
encoder -->|Nx2048|loss
mlp --> |Nx2048|loss
```