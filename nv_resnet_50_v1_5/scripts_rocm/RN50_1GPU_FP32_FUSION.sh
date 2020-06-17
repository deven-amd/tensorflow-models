# export MIOPEN_ENABLE_LOGGING_CMD=1

# export TF_ROCM_FMA_DISABLE=1
export TF_ROCM_FUSION_ENABLE=1

DATA_DIR=/data

RESULTS_DIR=results_FP32_FUSION
rm -rf ${RESULTS_DIR} && mkdir ${RESULTS_DIR}

python3 main.py \
       --mode=train_and_evaluate \
       --iter_unit=batch \
       --num_iter=100 \
       --batch_size=256 \
       --warmup_steps=10 \
       --use_cosine_lr \
       --label_smoothing 0.1 \
       --lr_init=0.256 \
       --lr_warmup_epochs=8 \
       --momentum=0.875 \
       --weight_decay=3.0517578125e-05 \
       --data_dir=${DATA_DIR} \
       --results_dir=${RESULTS_DIR}
