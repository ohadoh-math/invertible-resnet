#!/usr/bin/env bash

set -eux

output_dir=${OUTPUT_DIR:-results}
coeff=${COEFF:-0.9}
trunc=${TRUNC:-1}

if [ "${trunc}" -lt 1 ]
then
    truncated="_truncated"
fi

savedir="${output_dir}/cifar_coeff_${coeff}${truncated:-}_$(date +'%d.%m-%H%M')"
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
    --design-batch-size ${DESIGN_BATCH_SIZE:-20} \
    --vis_port 8097 &> >(tee ${savedir}/log)
