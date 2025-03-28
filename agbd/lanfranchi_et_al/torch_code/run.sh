#!/bin/bash

#SBATCH -n 10
#SBATCH --array=1-5
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --gpus=1
#SBATCH --tmp=6G

#SBATCH -o /cluster/work/igp_psr/albecker/shadetrees/biomass/logs/%J.txt
#SBATCH -e /cluster/work/igp_psr/albecker/shadetrees/biomass/logs/%J.txt


data_path=/cluster/work/igp_psr/albecker/shadetrees/biomass/final_all

cp ${data_path}/agbd_train.h5 $TMPDIR/
cp ${data_path}/agbd_val.h5 $TMPDIR/
cp ${data_path}/agbd_se_train.h5 $TMPDIR/
cp ${data_path}/agbd_se_val.h5 $TMPDIR/
cp ${data_path}/lat_train.h5 $TMPDIR/
cp ${data_path}/lat_val.h5 $TMPDIR/
cp ${data_path}/lon_train.h5 $TMPDIR/
cp ${data_path}/lon_val.h5 $TMPDIR/
cp ${data_path}/canopy_height_train.h5 $TMPDIR/
cp ${data_path}/canopy_height_val.h5 $TMPDIR/
cp ${data_path}/standard_deviation_train.h5 $TMPDIR/
cp ${data_path}/standard_deviation_val.h5 $TMPDIR/
cp ${data_path}/sample_weights.json $TMPDIR/
cp ${data_path}/normalization_values.csv $TMPDIR/

MODEL_IDX=$SLURM_ARRAY_TASK_ID

# model_path=/cluster/work/igp_psr/albecker/shadetrees/biomass/checkpoints

# Change the following parameters
arch="fcn_6_gaussian"
loss="WeightedLaplacianNLL"
sample_weighting_method="ifns"
use_nb_of_classes="true"
beta=0.999
latlon="false"
lat="true"
include_std="true"
n_epochs=150
batch_size=128
lr=0.00001
downsample="average"

if [ "$latlon" == "true" ]; then
    if [ "$include_std" == "true" ]; then
        in_features=5
    else
        in_features=4
    fi
elif [ "$lat" == "true" ]; then
    if [ "$include_std" == "true" ]; then
        in_features=3
    else
        in_features=2
    fi
else
    in_features=1
fi

echo $in_features

# model_name=${model_path}/best_${arch}_averagepool_${MODEL_IDX}_GaussianNLL_100000_${in_features}_0.0001_256_latOnly.pth

workdir=/cluster/work/igp_psr/albecker/shadetrees/biomass/code/torch_code

python ${workdir}/train.py --dataset_path $TMPDIR \
                --normalize_input "true" --normalize_gt "true" \
                --arch $arch \
                --loss_key $loss \
                --latlon $latlon \
                --lat $lat \
                --include_std $include_std \
                --downsample $downsample \
                --n_epochs $n_epochs \
                --batch_size $batch_size \
                --learning_rate $lr \
                --model_idx $MODEL_IDX \
                --use_nb_of_classes $use_nb_of_classes \
                --sample_weighting_method $sample_weighting_method \
		--tag all_w2_backup \
		--wd 0.00001
                #--n_iter 10000 \
