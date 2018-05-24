The Pilot1 Benchmark 4, commonly referred to as NT3, is a 1D convolutional network for classifying RNA-seq gene expression profiles into normal or tumor tissue categories. 
The network follows the classic architecture of convolutional models with multiple 1D convolutional layers interleaved with pooling layers followed by final dense layers. 
The network can optionally use 1D locally connected layers in place of convolution layers as well as dropout layers for regularization. 
The model is trained on the balanced 700 matched normal-tumor gene expression profile pairs available from the NCI genomic data commons. 
The full set of expression features contains 60,483 float columns transformed from RNA-seq FPKM-UQ values. This model achieves around 98% classification accuracy. 
It is useful for studying the difference and transformation of latent representation between normal and tumor tissues. 
The model also acts as a quality control check for synthetically generated gene expression profiles.
