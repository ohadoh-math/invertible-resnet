#!/usr/bin/env bash

set -eux

THIS_SCRIPT=$(readlink -f ${BASH_SOURCE[0]})
REPO_BASE=$(dirname $(dirname ${THIS_SCRIPT}))

output_dir="${REPO_BASE}/design-experiments/design-$(date +'%d.%m-%H%M')"
mkdir -p "${output_dir}"

cd ${REPO_BASE}

for coeff in 0.{5..9}
do
    for design in "uniform" "k-centers"
    do
        for iter in $(seq 1 ${ITERATIONS:-5})
        do
            savedir="${output_dir}/results-${design}-c${coeff}-i${iter}/"
            mkdir -p ${savedir}

            python3 ./CIFAR_main.py \
                --nBlocks 7 7 7 \
                --nStrides 1 2 2 \
                --nChannels 32 64 128 \
                --coeff ${coeff} \
                --batch 128 \
                --dataset cifar10 \
                --init_ds 1 \
                --inj_pad 13 \
                --powerIterSpectralNorm 1 \
                --save_dir ${savedir} \
                --nonlin elu \
                --optimizer sgd \
                --vis_server localhost \
                --design-epochs 10 \
                --train-epochs 20 \
                --design ${design} \
                --design-batch-size ${DESIGN_BATCH_SIZE:-64} \
                --vis_port 8097 &> >(tee ${savedir}/log)
        done
    done
done
