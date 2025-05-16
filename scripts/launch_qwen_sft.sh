#!/bin/bash
#SBATCH --job-name=gritkto
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a3low
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 30-00:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=/data/niklas/jobs/%x-%j.out           # output file name

BETA=$1
LR=$2

# Function to find an available port
find_free_port() {
    local port
    while true; do
        # Generate a random port number between 20000 and 65000
        port=$(shuf -i 29500-29510 -n 1)
        # Check if the port is in use
        if ! netstat -tuln | grep -q ":$port "; then
            echo "$port"
            break
        fi
    done
}

# Function to initialize the environment and print diagnostic information
# very important that this is run within srun for training to work!!!
init_env() {
    # Load necessary modules (adjust as needed for your system)
    # module load anaconda3/2024.2

    # Activate your conda environment
    # source $(conda info --base)/etc/profile.d/conda.sh
    source /env/bin/start-ctx-user
    conda activate gritkto

    echo "Running on node: $(hostname)"
    echo "Machine Rank: $SLURM_PROCID"
    
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=$(find_free_port | tr -d '\n')
    # export HF_DATASETS_OFFLINE=1
    # export HF_HUB_OFFLINE=1
    
    echo "Master node: $MASTER_ADDR"
    echo "Number of nodes: $SLURM_JOB_NUM_NODES"
    echo "GPUs per node: $SLURM_GPUS_PER_NODE"
}

export -f find_free_port
export -f init_env

# Run the training script using srun
srun --jobid=$SLURM_JOB_ID --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "
init_env
export MODEL_PATH=Qwen/Qwen2.5-32B-Instruct
export CKPT=/data/niklas/models/qwen-32b-sft

accelerate launch \
    --config_file accelerate_config/fsdp_2x8gpu.yaml \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    launch.py loss=sft model=qwen train_datasets=[s1k_11] test_datasets=[s1k_11] exp_name=qwen-32B-sft n_epochs=5 warmup_steps=16 \
    ++cache_dir=/data/niklas/models \
    ++model.name_or_path=\$MODEL_PATH \
    ++model.use_peft=false ++cache_reference_logprobs=false ++lr=1e-6 \
    ++model.batch_size=32 ++model.gradient_accumulation_steps=1 ++model.micro_batch_size=1 \
    ++model.max_length=32768 ++model.max_prompt_length=32768 \
    ++model.attn_implementation=flash_attention_2

python -m train.sample \$CKPT --gpu_count 2 --output_file outputs/qwen-32b-sft.json
"