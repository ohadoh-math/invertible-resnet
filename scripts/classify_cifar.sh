#!/usr/bin/env bash
set -eux
BATCH_SIZE=32 # 128
python3 ./CIFAR_main.py \
    --nBlocks 7 7 7 \
    --nStrides 1 2 2 \
    --nChannels 32 64 128 \
    --coeff 0.9 \
    --batch ${BATCH_SIZE} \
    --dataset cifar10 \
    --init_ds 1 \
    --inj_pad 13 \
    --powerIterSpectralNorm 1 \
    --save_dir ./results/zca_clf_full_cifar10_wrn22_inj_pad_coeff09 \
    --nonlin elu \
    --optimizer sgd \
    --vis_server localhost \
    --vis_port 8097
