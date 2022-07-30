Run the multi worker example like this:

```
CUDA_VISIBLE_DEVICES=0,1 python ./multi_worker.py 0
CUDA_VISIBLE_DEVICES=2,3 PYTHON ./multi_worker.py 1
```
