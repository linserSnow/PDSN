#!/usr/bin/env bash
PROJECT="/home/lingxuesong/projects/PDSN/"
cd "$PROJECT"
function runExtract()
{
    python extract_mask_dic_mean.py \
        --train_list=/home/lingxuesong/data/VGG-Face1/sia/lists_calm/pair_same_calm_${1}_extra.txt \
        --weight_model=checkpoint/dicts/d${1}/n2_w2/CosFace_${2}_checkpoint.pth \
        --save_path=masks/mean/d${1}/n2w2 \
        --gpus=0 \
        --ngpus=1
}
runExtract 12 49


