#!/usr/bin/env bash
set -eux

coeff=${COEFF:-0.9}
trunc=${TRUNC:-1}

if [ "${trunc}" -lt 1 ]
then
    truncated="_truncated"
fi

savedir="results/cifar_coeff_${coeff}${truncated:-}_$(date +'%d.%m-%H%M')"
mkdir -p ${savedir}

python3 ./CIFAR_main.py \
    --nBlocks 7 7 7 \
    --nStrides 1 2 2 \
    --nChannels 32 64 128 \
    --coeff ${coeff} \
    --batch ${BATCH_SIZE:-128} \
    --dataset ${DATASET:-cifar10} \
    --init_ds 1 \
    --inj_pad 13 \
    --powerIterSpectralNorm ${POWER_ITER_SPEC_NORM:-1} \
    --save_dir ${savedir} \
    --nonlin elu \
    --optimizer sgd \
    --vis_server localhost \
    --design-epochs ${DESIGN_EPOCHS:-10} \
    --train-epochs ${TRAIN_EPOCHS:-20} \
    --trunc ${trunc} \
    --design ${DESIGN:-none} \
    --vis_port 8097 &> >(tee ${savedir}/log)
