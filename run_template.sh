#!/usr/bin/env bash

nohup python run.py \
  --prefix exp_stage_1 \
  --mixture_dim 2 8 32 128 \
  --n_components_max 5 \
  --n_components_min 2 \
  --n_embd 128 256 512 \
  --n_layer 3 6 12 24 \
  --train_n_sample 32 64 128 \
  --eval_n_sample 32,64,128 \
  --num_train_steps 100001 \
 > run.log 2>&1 &