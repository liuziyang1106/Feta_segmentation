#!/bin/bash
csize=64,64,64
save_path=./runs/3DUnet_base_20_dice_loss_64*64*64/
CUDA_VISIBLE_DEVICES=0     python train.py     \
--output_dir          ${save_path}             \
--batch_size          8                        \
--epochs              100                      \
--lr                  1e-3                     \
--crop_size           ${csize}                 \
--print_freq          10                       \
