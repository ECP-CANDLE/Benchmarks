The Pilot1 Benchmark 5, commonly referred to as TC1, is a 1D convolutional network for classifying RNA-seq gene expression profiles into 18 balanced tumor types (e.g., breast cancer, melanoma, etc). 
The network follows the classic architecture of convolutional models with multiple 1D convolutional layers interleaved with pooling layers followed by final dense layers. 
The network can optionally use 1D locally connected layers in place of convolution layers as well as dropout layers for regularization. 
The model is trained and cross-validated on a total of 5,400 RNA-seq profiles from the NCI genomic data commons. 
The full set of expression features contains 60,483 float columns transformed from RNA-seq FPKM-UQ values. This model achieves around 98% classification accuracy. 
It is useful for studying the relationships between latent representations of different tumor types as well as classifying synthetically generated gene expression profiles. 
The model has also been used to flag incorrectly typed gene expression profiles from the databases
