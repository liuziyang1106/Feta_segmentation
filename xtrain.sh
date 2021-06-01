#!/bin/bash
csize=96,128,96
tipath=./data/data_2.1/T2/
tmpath=./data/data_2.1/Seg/
ifpath=./data/data_2.1/Test/

save_path=./runs/3DUnet_base_FeTA2.1_40_dice_loss_96*128*96/

CUDA_VISIBLE_DEVICES=0     python train.py     \
--output_dir          ${save_path}             \
--batch_size          4                        \
--epochs              200                      \
--lr                  1e-3                     \
--crop_size           ${csize}                 \
--print_freq          10                       \
--train_img_folder    ${tipath}                \
--train_mask_folder   ${tmpath}                \
--test_img_folder     ${ifpath}                \
--lbd                 40                       \
