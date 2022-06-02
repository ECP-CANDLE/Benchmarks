# P3B8 BERT Quantized

Quantizing BERT on synthetic data that imitates the Mimic dataset.

Run locally:

```
python py3b8_baseline.py
```

Run on Summit:

```
bsub run.lsf
```

Training for 10 epochs:

```
epoch: 0, validation F1: 0.4353591160220994
epoch: 1, validation F1: 0.4353591160220994
epoch: 2, validation F1: 0.5792531120331951
epoch: 3, validation F1: 0.4975124378109453
epoch: 4, validation F1: 0.4975124378109453
epoch: 5, validation F1: 0.4975124378109453
epoch: 6, validation F1: 0.4975124378109453
epoch: 7, validation F1: 0.4975124378109453
epoch: 8, validation F1: 0.4975124378109453
epoch: 9, validation F1: 0.4975124378109453
```

Using dynamic quantization, we set BERT's parameters to int8,
reducing the model size by 82.8%:

|      Model     |   Size (MB)  |  Inference Time (s) |     F1      |
| :------------- | :----------: |  :----------------: | ----------: |
|  BERT FP32     |  438.043912  |        26.4         |   0.497512  |
|  BERT INT8     |  181.507765  |        27.8         |   0.497512  |

