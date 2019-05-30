The Pilot1 Benchmark 5, commonly referred to as TC1, is a 1D convolutional network for classifying RNA-seq gene expression profiles into 18 balanced tumor types (e.g., breast cancer, melanoma, etc). 
The network follows the classic architecture of convolutional models with multiple 1D convolutional layers interleaved with pooling layers followed by final dense layers. 
The network can optionally use 1D locally connected layers in place of convolution layers as well as dropout layers for regularization. 
The model is trained and cross-validated on a total of 5,400 RNA-seq profiles from the NCI genomic data commons. 
The full set of expression features contains 60,483 float columns transformed from RNA-seq FPKM-UQ values. This model achieves around 98% classification accuracy. 
It is useful for studying the relationships between latent representations of different tumor types as well as classifying synthetically generated gene expression profiles. 
The model has also been used to flag incorrectly typed gene expression profiles from the databases

## Profile runs
We have run the same configuration across multiple machines and compared the resource utilization. 
```
python uno_baseline_keras2.py --conf tc1_perf_benchmark.txt
```

| Machine | Time to complete (HH:mm:ss) | Time per epoch (s) | Perf factor <sup>*</sup> | CPU % | Mem % | Mem GB | GPU % | GPU Mem % | Note |
| ------- | --------------------------: | -----------------: | -----------------------: | ----: | ----: | -----: | ----: | --------: | ---- |
| Theta | | | | | | | | | <sup>1</sup> |
| Nucleus | 0:35:33 | 74 | 5.34 | 3.0 | 6.5 | 14.3 | 81.7 | 92.7 | |
| Tesla (K20) | 2:22:31 | 394 | 1.00 | 27.1 | 40.9 | 11.6 | 97.5 | 45.4 | |
| Titan | 1:55:39 | 395 | 1.00 | 5.4 | 28.5 | 7.1 | | | |
1. MKL-DNN does not support Conv1D. need tf1.11
* Time per epoch on the machine divided by time per epoch of Titan (or Tesla)