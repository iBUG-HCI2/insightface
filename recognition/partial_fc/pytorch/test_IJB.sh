#!/bin/bash
export CUDA_VISIBLE_DEVICES="4,5,6,7"
METHOD=partial_fc_original
echo $METHOD
work_dir=${PWD}
root=/home/yiminglin/insightface/datasets
GPU_ID=7
IJB=IJBB
result_dir="$root/result/$IJB/$METHOD"
echo $result_dir
cd ./IJB
echo $work_dir
python -u IJB_11_Batch.py --model-prefix $work_dir/models/19backbone.pth \
        --image-path ${root}/IJB_release/${IJB} \
        --result-dir ${result_dir} \
        --model-epoch 0 --gpu ${GPU_ID} \
        --target ${IJB} --job ${METHOD} \
        --batch-size 1024

IJB=IJBC
result_dir="$root/result/$IJB/$METHOD"
python -u IJB_11_Batch.py --model-prefix $work_dir/models/19backbone.pth \
        --image-path ${root}/IJB_release/${IJB} \
        --result-dir ${result_dir} \
        --model-epoch 0 --gpu ${GPU_ID} \
        --target ${IJB} --job ${METHOD} \
        --batch-size 1024

