#!/bin/bash

python ./bert_webdataset.py \
   --output_dir ./outputs \
   --do_train=True \
   --per_device_train_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --max_len 512 \
   --learning_rate 0.0001 \
   --adam_beta2 0.98 \
   --weight_decay 0.0000 \
   --adam_epsilon 2e-8 \
   --max_steps 10 \
   --warmup_steps 1
