#!/bin/bash
csize=64,64,128
tipath=/home/liuziyang/workspace/FeTA/Pytorch-UNet/data/img_resample_crop/
tmpath=/home/liuziyang/workspace/FeTA/Pytorch-UNet/data/mask_resample_crop/
ifpath=/home/liuziyang/workspace/FeTA/Pytorch-UNet/data/test_resample/

save_path=./runs/3DUnet_base_resample_40_dice_loss_64*64*128/

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
