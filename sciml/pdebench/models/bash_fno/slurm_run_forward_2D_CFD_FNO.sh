source ~/.bashrc
conda activate pdebench_fno
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
root=$(pwd)
## 'FNO'
SLURM_ARRAY_TASK_ID=$1
CONFIG_FILE=$2

cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${root}/pdebench/txt_files/fno_2dcfd.txt)
model_name=$(echo $cfg | cut -f 1 -d ' ')
batch_size=$(echo $cfg | cut -f 2 -d ' ')
filename=$(echo $cfg | cut -f 3 -d ' ')
epochs=$(echo $cfg | cut -f 4 -d ' ')
learning_rate=$(echo $cfg | cut -f 5 -d ' ')
modes=$(echo $cfg | cut -f 6 -d ' ')
width=$(echo $cfg | cut -f 7 -d ' ')
weight_decay=$(echo $cfg | cut -f 9 -d ' ') 
optimizer=$(echo $cfg | cut -f 10 -d ' ')
scheduler=$(echo $cfg | cut -f 11 -d ' ')
temp_balance_lr=$(echo $cfg | cut -f 12 -d ' ')
fix_fingers=$(echo $cfg | cut -f 13 -d ' ')
tb_metric=$(echo $cfg | cut -f 14 -d ' ')
lr_min_ratio=$(echo $cfg | cut -f 15 -d ' ')
lr_slope=$(echo $cfg | cut -f 16 -d ' ')
tb_interval_ep=$(echo $cfg | cut -f 17 -d ' ')
eigs_thres=$(echo $cfg | cut -f 18 -d ' ')
tb_batchnorm=$(echo $cfg | cut -f 19 -d ' ')
tb_hyperparam=$(echo $cfg | cut -f 21 -d ' ')
reduced_batch=$(echo $cfg | cut -f 22 -d ' ')
exp_temp=$(echo $cfg | cut -f 23 -d ' ')

for random_seed in 2024 #2023 2022
        do
                echo ................Start........... 
                python pdebench/models/train_models_forward.py \
                        +args=${CONFIG_FILE} \
                        ++args.model_name=${model_name} \
                        ++args.batch_size=${batch_size} \
                        ++args.filename=${filename} \
                        ++args.epochs=${epochs} \
                        ++args.learning_rate=${learning_rate} \
                        ++args.modes=${modes} \
                        ++args.width=${width} \
                        ++args.random_seed=${random_seed} \
                        ++args.weight_decay=${weight_decay} \
                        ++args.optimizer=${optimizer} \
                        ++args.scheduler=${scheduler} \
                        ++args.temp_balance_lr=${temp_balance_lr} \
                        ++args.fix_fingers=${fix_fingers} \
                        ++args.tb_metric=${tb_metric} \
                        ++args.lr_min_ratio=${lr_min_ratio} \
                        ++args.lr_slope=${lr_slope} \
                        ++args.tb_interval_ep=${tb_interval_ep} \
                        ++args.eigs_thres=${eigs_thres} \
                        ++args.tb_batchnorm=${tb_batchnorm} \
                        ++args.tb_hyperparam=${tb_hyperparam} \
                        ++args.reduced_batch=${reduced_batch} \
                        ++args.task_id=${SLURM_ARRAY_TASK_ID} \
                        ++args.exp_temp=${exp_temp}
                echo ................End...........
        done

    
    


