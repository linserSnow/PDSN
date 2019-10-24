#!/bin/bash
PROJECT="/home/lingxuesong/projects/PDSN/"
cd "$PROJECT"
function runTrainDic()
{
    python -u main_dic.py \
        --train_list=/home/lingxuesong/data/sia/lists/train_pair_occ_${1}_extra.txt \
        --valid_list=/home/lingxuesong/data/sia/lists/valid_pair_occ_${1}.txt \
        -c=1 \
        --gpus=0,1,2,3 \
        --ngpus=4 \
        --save_path=checkpoint/dicts/d${1}/n2w2 \
        --d_name=d${1} \
        2>&1 | tee ./log/dict/cosface_sia_d${1}_n2w2_`date +%Y%m%d%H%M`.log
}
runTrainDic 12
