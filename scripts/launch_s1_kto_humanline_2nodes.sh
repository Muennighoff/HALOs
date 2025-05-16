#!/bin/bash
#SBATCH --job-name=gritkto
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a3low
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 30-00:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=/data/niklas/jobs/%x-%j.out           # output file name

source /env/bin/start-ctx-user
conda activate gritkto

GPUS_PER_NODE=8
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

MODEL_PATH=Qwen/Qwen2.5-32B-Instruct
CKPT=/data/niklas/models/s1.1-32b-kto-humanline-v2

# Run the training script using srun
srun --jobid=$SLURM_JOB_ID --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "
accelerate launch \
    --config_file accelerate_config/fsdp_2x8gpu.yaml \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    --num_processes $WORLD_SIZE \
    --role $SLURMD_NODENAME: \
    --rdzv_conf rdzv_backend=c10d \
    --max_restarts 0 \
    launch.py loss=kto model=qwen train_datasets=[s1k_11] test_datasets=[s1k_11] exp_name=s1.1-32B-kto-humanline-v2 do_first_eval=false \
    ++cache_dir=/data/niklas/models \
    ++model.name_or_path=$MODEL_PATH \
    ++model.batch_size=16 ++model.gradient_accumulation_steps=1 ++model.eval_batch_size=16 \
    ++model.max_length=10000 ++model.max_prompt_length=10000 \
    ++model.attn_implementation=flash_attention_2 ++eval_every=999999 ++n_eval_examples=0 ++humanline=true

python -m train.sample $CKPT --gpu_count 2 --output_file outputs/qwen-32b-kto-humanline.json
"