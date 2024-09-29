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

temp_balance_lr=tb_layer_blocks
temp_balance_wd=tb_layer_blocks
metric=alpha
fix_fingers=xmin_mid
xmin_pos=1.5
fix_tb_metric=False
remove_last_layer=True

for SLURM_ARRAY_TASK_ID in {3..20..1}
    do
        cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p $(pwd)/config_files/configs_sigmoid.txt)
        TASK_NAME=$(echo $cfg | cut -f 1 -d ' ')
        LR=$(echo $cfg | cut -f 2 -d ' ')
        batch_size=$(echo $cfg | cut -f 3 -d ' ')
        seed=$(echo $cfg | cut -f 4 -d ' ')
        NUM_EPOCHS=$(echo $cfg | cut -f 5 -d ' ')
        hyperparam_to_tune=$(echo $cfg | cut -f 6 -d ' ')
        tb_unit=$(echo $cfg | cut -f 7 -d ' ')
        esd_type=$(echo $cfg | cut -f 8 -d ' ')
        schedule_func=$(echo $cfg | cut -f 9 -d ' ')
        tb_lr_normalize=$(echo $cfg | cut -f 10 -d ' ')
        after_warm=$(echo $cfg | cut -f 11 -d ' ')
        data_ratio=$(echo $cfg | cut -f 12 -d ' ')
        sigmoid_min=$(echo $cfg | cut -f 13 -d ' ')
        sigmoid_max=$(echo $cfg | cut -f 14 -d ' ')
        fix_fingers=$(echo $cfg | cut -f 15 -d ' ')

        OUTPUT_DIR=$CKPT_SRC_DIR/finetune/$MODEL_NAME/$TASK_NAME/tempbalance_ratio_${data_ratio}/lr_${LR}_epochs_${NUM_EPOCHS}_bs_${batch_size}_warmup_${warmup}_wd_${weight_decay}_max_seq_${max_seq_length}/${esd_type}_tb_${metric}_${schedule_func}_${temp_balance_lr}_unit_${tb_unit}_metric_${metric}_fix_${fix_fingers}_xmin_pos_${xmin_pos}_lrnorm_${tb_lr_normalize}_fix_metric_${fix_tb_metric}_remove_last_${remove_last_layer}_after_warm_${after_warm}/tune_${hyperparam_to_tune}_sigmoid_min_${sigmoid_min}_max_${sigmoid_max}/seed_${seed};

        mkdir -p $OUTPUT_DIR/metrics_stats

        cd $SRC_DIR/examples/pytorch/text-classification

        python run_glue.py \
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
            --logging_dir $OUTPUT_DIR \
            --overwrite_output_dir \
            --use_tb \
            --after_warm $after_warm \
            --hyperparam_to_tune $hyperparam_to_tune \
            --schedule_func $schedule_func \
            --fix_tb_metric $fix_tb_metric \
            --temp_balance_lr $temp_balance_lr \
            --temp_balance_wd $temp_balance_wd \
            --tb_lr_normalize $tb_lr_normalize \
            --remove_last_layer $remove_last_layer \
            --tb_unit $tb_unit \
            --metric $metric \
            --esd_type $esd_type \
            --fix_fingers $fix_fingers \
            --xmin_pos $xmin_pos \
            --warmup_ratio $warmup \
            --weight_decay $weight_decay \
            --save_total_limit 1 \
            --report_to none \
            --seed $seed \
            --train_data_ratio $data_ratio \
            --sigmoid_min $sigmoid_min \
            --sigmoid_max $sigmoid_max
    done