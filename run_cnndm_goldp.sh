TOTAL_NUM_UPDATES=200000
WARMUP_UPDATES=500
LR=2e-05
MAX_TOKENS=1024  ## 2048 if machine permits
UPDATE_FREQ=2
BART_PATH=[TODO]
MLE_PATH=[TODO]

## CUDA_VISIBLE_DEVICES=0,1,2,3
## Potentially use python -W ignore
python train.py cnn_dm-bin \
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
    --save-interval-updates 10000 --keep-interval-updates 3 --p40 1 --reset-optimizer \
    --fp16  # can drop fp16; note that the default reward-type is logp
