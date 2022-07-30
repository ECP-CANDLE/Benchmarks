Run the multi worker example like this:

```
CUDA_VISIBLE_DEVICES=0,1 python ./multi_worker.py 0 &
CUDA_VISIBLE_DEVICES=2,3 python ./multi_worker.py 1 &
```

Output will be written to a log file
