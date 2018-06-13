#!/bin/bash
# --checkpoint_path=${HOME}/features_filtered/augmented/model \
CUDA_VISIBLE_DEVICES=0 python eval_image_classifier.py \
  --alsologtostderr \
  --eval_dir=/tmp/tfmodel \
  --dataset_name=patches \
  --model_name=inception_v3 \
  --checkpoint_path=/home/jayant/features_filtered/augmented/scratch_5e3 \
  --dataset_dir=${HOME}/features_filtered/handpicked \
  --dataset_split_name=test \
  # --checkpoint_path=/home/jayant/features_filtered/handpicked/model/model.ckpt-186829 \
  # --checkpoint_path=/home/jayant/features_filtered/augmented/scratch_5e3 \
  # --checkpoint_path=/home/jayant/features_filtered/augmented_conf_optflow_1/model \
