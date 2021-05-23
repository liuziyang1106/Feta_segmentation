#!/bin/bash

save_path=./runs/test_dice_loss_64*100*100/
CUDA_VISIBLE_DEVICES=0     python train.py     \
--output_dir          ${save_path}             \
--batch_size          1                        \
--epochs              100                      \
--lr                  1e-3                     \
--print_freq          10                       \