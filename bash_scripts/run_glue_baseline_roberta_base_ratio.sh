#!/bin/bash

export PYTHONUNBUFFERED=1
export HF_HOME=$(pwd)/huggingface
source ~/.bashrc
source activate base
conda activate ww_finetune


SRC_DIR=$(pwd)/transformers
CKPT_SRC_DIR=$(pwd)/checkpoints/nlp
MODEL_NAME=FacebookAI/roberta-base
warmup=0.06
weight_decay=0.1
max_seq_length=128
fix_fingers=xmin_mid
xmin_pos=1.5

for SLURM_ARRAY_TASK_ID in {3..20..1}
    do
        cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p $(pwd)/config_files/configs.txt)
        TASK_NAME=$(echo $cfg | cut -f 1 -d ' ')
        LR=$(echo $cfg | cut -f 2 -d ' ')
        batch_size=$(echo $cfg | cut -f 3 -d ' ')
        seed=$(echo $cfg | cut -f 4 -d ' ')
        NUM_EPOCHS=$(echo $cfg | cut -f 5 -d ' ')
        data_ratio=$(echo $cfg | cut -f 6 -d ' ')

        OUTPUT_DIR=$CKPT_SRC_DIR/finetune/$MODEL_NAME/$TASK_NAME/baseline_ratio_${data_ratio}/lr_${LR}_epochs_${NUM_EPOCHS}_bs_${batch_size}_warmup_${warmup}_wd_${weight_decay}_max_seq_${max_seq_length}/fix_${fix_fingers}_xmin_pos_${xmin_pos}/seed_${seed}

        mkdir -p $OUTPUT_DIR/metrics_stats

        cd $SRC_DIR/examples/pytorch/text-classification

        CUDA_VISIBLE_DEVICES=0 python run_glue.py \
            --model_name_or_path $MODEL_NAME \
            --task_name $TASK_NAME \
            --do_train \
            --do_eval \
            --max_seq_length $max_seq_length \
            --per_device_train_batch_size $batch_size \
            --learning_rate $LR \
            --num_train_epochs $NUM_EPOCHS \
            --logging_steps 10 \
            --evaluation_strategy epoch \
            --output_dir $OUTPUT_DIR \
            --fix_fingers $fix_fingers \
            --xmin_pos $xmin_pos \
            --overwrite_output_dir \
            --warmup_ratio $warmup \
            --weight_decay $weight_decay \
            --save_total_limit 1 \
            --report_to none \
            --seed $seed \
            --train_data_ratio $data_ratio
    done