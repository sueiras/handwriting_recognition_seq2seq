#!/bin/bash
DATA_PATH=./data

nohup python seq2seq_evaluate.py \
  --cuda_device '0' \
  --experiment 'IAM/BILSTM2_256_sz10_st_2' \
  --dataset 'IAM_words_48_192.hdf5' \
  --slides_stride 2 \
  --x_slide_size 10 \
  --bidirectional \
  --num_layers 2 \
  --dim_lstm 256 \
  --num_heads 1 \
  --keep_prob 0.5 \
  --batch_size 256 > ${DATA_PATH}/IAM/BILSTM2_256_sz10_st_2_eval.out &
