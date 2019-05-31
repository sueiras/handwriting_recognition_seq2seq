#!/bin/bash

export DATA_PATH=./data

nohup python seq2seq_train.py \
  --cuda_device '1' \
  --experiment 'IAM/BILSTM2_256_sz10_st_2' \
  --train_data 'IAM_words_48_192.hdf5' \
  --nouse_pretrained_char_model \
  --pct_lr_char_model 1.0 \
  --slides_stride 2 \
  --x_slide_size 10 \
  --bidirectional \
  --num_layers 2 \
  --dim_lstm 256 \
  --num_heads 1 \
  --lambda_l2_reg 0.0001 \
  --learning_rate 0.001 \
  --exponential_decay_step 400 \
  --exponential_decay_rate 0.98 \
  --keep_prob 0.5 º
  --batch_size 256 > ${DATA_PATH}/IAM/BILSTM2_256_sz10_st_2.out &
