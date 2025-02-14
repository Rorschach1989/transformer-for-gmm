#!/usr/bin/env bash

nohup python run.py \
  --prefix exp_stage_1 \
  --mixture_dim 2 4 8 16 32 64 128 256 \
  --n_components_max 3 4 5 \
  --n_components_min 2 \
  --n_embd 128 256 512 1024 \
  --n_layer 12 24 36 \
  --train_n_sample 32 64 128 \
  --eval_n_sample 32 64 128 \
  --num_train_steps 100001 \
 > run.log 2>&1 &