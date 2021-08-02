The Pilot1 Benchmark 4, commonly referred to as NT3, is a 1D convolutional network for classifying RNA-seq gene expression profiles into normal or tumor tissue categories. 
The network follows the classic architecture of convolutional models with multiple 1D convolutional layers interleaved with pooling layers followed by final dense layers. 
The network can optionally use 1D locally connected layers in place of convolution layers as well as dropout layers for regularization. 
It is useful for studying the difference and transformation of latent representation between normal and tumor tissues. 
The model also acts as a quality control check for synthetically generated gene expression profiles.

## Data
The model is trained on the balanced 700 matched normal-tumor gene expression profile pairs available from the NCI genomic data commons. 
The full set of expression features contains 60,483 float columns from RNA-seq [FPKM-UQ](https://docs.gdc.cancer.gov/Encyclopedia/pages/HTSeq-FPKM-UQ/) values. This model achieves around 98% classification accuracy. 
The associated metadata for the samples can be found [here](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/normal-tumor/matched_normal_samples.metadata.tsv). 

## Profile runs
We have run the same configuration across multiple machines and compared the resource utilization. 
```
python nt3_baseline_keras2.py --conf nt3_perf_bench_model.txt
```

| Machine | Time to complete (HH:mm:ss) | Time per epoch (s) | Perf factor <sup>*</sup> | CPU % | Mem % | Mem GB | GPU % | GPU Mem % | Note |
| ------- | --------------------------: | -----------------: | -----------------------: | ----: | ----: | -----: | ----: | --------: | ---- |
| Theta | | | | | | | | | <sup>1</sup> |
| Nucleus (V100) | 0:10:38 | 11 | 9.00 | 7.9 | 19.9 | 48.5 | 75.3 | 87.2 | |
| Tesla (K20) | 1:29:06 | 103 | 0.96 | 18.1 | 25.8 | 6.8 | 97.3 | 45.3 | |
| Titan | 1:07:31 | 99 | 1.00 | 5.1 | 15.5 | 3.8 | | | |
1. MKL-DNN does not support Conv1D. need tf1.11
* Time per epoch on the machine divided by time per epoch of Titan (or Tesla)
