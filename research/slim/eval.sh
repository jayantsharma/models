#!/bin/bash
# --checkpoint_path=${HOME}/features_filtered/augmented/model \
CUDA_VISIBLE_DEVICES=2 python eval_image_classifier.py \
  --alsologtostderr \
  --checkpoint_path=/home/jayant/features_filtered/augmented_conf_optflow_1/model \
  --eval_dir=/home/jayant/features_filtered/augmented_conf_optflow_1/eval \
  --dataset_dir=${HOME}/features_filtered/augmented_conf_optflow_1 \
  --dataset_name=patches \
  --dataset_split_name=test \
  --model_name=inception_v3
