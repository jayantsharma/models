#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_image_classifier.py \
  --train_dir=/home/jayant/features_filtered/augmented_conf_optflow_1/model \
  --checkpoint_path=/home/jayant/my-checkpoints/inception_v3.ckpt \
  --dataset_dir=/home/jayant/features_filtered/augmented_conf_optflow_1 \
  --dataset_name=patches \
  --dataset_split_name=train \
  --model_name=inception_v3 \
  --save_interval_secs=300 \
  --save_summaries_secs=300 \
  --learning_rate=1e-2 \
  --num_epochs_per_decay=40 \
  --end_learning_rate=1e-7 \
  --batch_size=128 \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=150000 \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  ## EXCLUDE WHEN ITERATING ON CLASSIFIER ON SAME DATASET ##
