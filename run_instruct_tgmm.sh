TGMM_BACKBONE_CKPT=''

torchrun --nproc_per_node=2 \
  run_instruct_hf.py \
  --output_dir ./output \
  --report_to none \
  --label_names mixture_probs assignment gaussian_means scale \
  --max_steps 10000 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 128 \
  --learning_rate 5e-5 \
  --logging_steps 50 \
  --eval_strategy steps \
  --eval_steps 200 \
  --save_strategy steps \
  --save_total_limit 1 \
  --save_only_model true \
  --save_steps 1000 \
  --gradient_accumulation_steps 2 \
  --tgmm_backbone_ckpt_path $TGMM_BACKBONE_CKPT \
  --tgmm_components 2 3 4