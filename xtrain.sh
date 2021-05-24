#!/bin/bash
csize=64,64,128
save_path=./runs/test_dice_loss_64*64*128/
CUDA_VISIBLE_DEVICES=0     python train.py     \
--output_dir          ${save_path}             \
--batch_size          2                        \
--epochs              100                      \
--lr                  1e-3                     \
--crop_size           ${csize}                 \
--print_freq          10                       \