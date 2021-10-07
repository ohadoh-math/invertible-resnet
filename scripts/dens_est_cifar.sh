#!/usr/bin/env bash


python CIFAR_main.py  \
    --nBlocks 16 16 16  \
    --nStrides 1 2 2  \
    --nChannels 512 512 512  \
    --coeff 0.9 -densityEstimation -multiScale  \
    --lr 0.003  \
    --weight_decay 0.  \
    --numSeriesTerms 5  \
    --dataset cifar10  \
    --batch 128  \
    --warmup_epochs 1  \
    --save_dir ./results/dens_est_cifar  \
    --epochs "${EPOCHS:-10}" \
    --vis_server 127.0.0.1  \
    --vis_port 8097
