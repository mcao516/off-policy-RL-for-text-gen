#!/bin/bash
#SBATCH --account=def-jcheung            # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=3:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/<user>/slurm-%j.out  # Write the log in $SCRATCH

# 1. Create your environement locally
# module load miniconda3
# source activate alan
source ~/envRL/bin/activate

TOTAL_NUM_UPDATES=150000
WARMUP_UPDATES=1000
LR=2e-05
MAX_TOKENS=2048 ## 2048
UPDATE_FREQ=1

BART_PATH=/home/mcao610/scratch/BART_models/bart.large.xsum/model.pt
MLE_PATH=/home/mcao610/scratch/BART_models/bart.large.xsum/model.pt
DATA_PATH=/home/mcao610/scratch/summarization/XSum/fairseq_files/xsum-bin
SAVE_PATH=/home/mcao610/scratch/BART_models/checkpoints_xsum_golds_zeros_padding
mkdir $SAVE_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/mcao610/gold-off-policy-text-gen-iclr21/train.py $DATA_PATH \
    --save-dir $SAVE_PATH \
    --restore-file $BART_PATH \
    --load-path-mle $MLE_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --save-interval-updates 10000 --keep-interval-updates 3 --p40 1 --policy-update-per-k-epoch 5000 \
    --q-baseline -10.0 --iw-min 0.15 --reward-type sump --trunc-min 0.1 --reset-optimizer \
    # --find-unused-parameters;