#!/bin/bash

save_path=./runs/test_64*128*128/
CUDA_VISIBLE_DEVICES=0     python train.py     \
--output_dir          ${save_path}             \
--batch_size          2                        \
--epochs              100                      \
--lr                  1e-3                     \
--print_freq          10                        \