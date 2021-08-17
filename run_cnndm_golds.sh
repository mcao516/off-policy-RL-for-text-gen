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

TOTAL_NUM_UPDATES=200000
WARMUP_UPDATES=500
LR=2e-05
MAX_TOKENS=2048  ## 2048 if hardware permits
UPDATE_FREQ=2
BART_PATH=/home/mcao610/scratch/BART_models/bart.large.cnn/model.pt
MLE_PATH=/home/mcao610/scratch/BART_models/bart.large.cnn/model.pt
DATA_PATH=/home/mcao610/scratch/summarization/CNNDM/fairseq_files/cnn_dm-bin/

## CUDA_VISIBLE_DEVICES=0,1,2,3
## Potentially use python -W ignore
python train.py $DATA_PATH \
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
    --find-unused-parameters \
    --save-interval-updates 10000 --keep-interval-updates 3 --p40 1 --iw-min 0.1 --reward-type sump --trunc-min 0.0 \
    --reset-optimizer \
    --fp16  ## can drop fp16
