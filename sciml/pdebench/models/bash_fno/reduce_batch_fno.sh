pwd
hostname
date
echo starting test job...
source ~/.bashrc

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
root=$(pwd)


config_file=config_2DCFD_TB.yaml
for line in 2 3 4
    do
        CUDA_VISIBLE_DEVICES=##GPU ID   ./pdebench/models/bash_fno/slurm_run_forward_2D_CFD_FNO.sh ${line} ${config_file}
    done
    



