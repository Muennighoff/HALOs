#!/bin/bash
# example usage:
# bash scripts/debug_qwen_8gpu_grpo_async.sh 0.05 1e-6 0.2

set -x

KL_COEF=$1 # 0.05
LR=$2 # 1e-6
CLIP_RANGE=$3 # 0.2
NUM_GPUS=8

TOTAL_PROMPTS=10240
PROMPTS_PER_ROUND=1024  # Train on fewer examples before sampling
NUM_ROUNDS=$(($TOTAL_PROMPTS / $PROMPTS_PER_ROUND))  # Calculate this once at the start

# Function to find an available port
find_free_port() {
    local port
    while true; do
        port=$(shuf -i 29500-29510 -n 1)
        if ! netstat -tuln | grep -q ":$port "; then
            echo "$port"
            break
        fi
    done
}

# Function to initialize the environment
init_env() {
    # module load anaconda3/2024.2
    # source $(conda info --base)/etc/profile.d/conda.sh
    # conda activate halos
    source /env/bin/start-ctx-user
    conda activate gritkto

    echo "Running on node: $(hostname)"
    echo "Machine Rank: $SLURM_PROCID"
    
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=$(find_free_port | tr -d '\n')
    # export HF_DATASETS_OFFLINE=1
    # export HF_HUB_OFFLINE=1
    export HF_ALLOW_CODE_EVAL=1
}

export -f find_free_port
export -f init_env

# Running the training script
init_env
export MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
export HALO_DIR=/data/niklas/HALOs
mkdir -p $HALO_DIR/logs
export CACHE_DIR=/data/niklas/HALOs/models/qwen15-online-grpo-kl-${KL_COEF}-lr-${LR}-clip-${CLIP_RANGE}-deepscaler

if [ ! -d "${CACHE_DIR}" ]; then
    mkdir -p "${CACHE_DIR}"
fi

# Start iterative training from SFT checkpoint or last saved checkpoint
CURRENT_CKPT=$MODEL_PATH
ROUND=1
CUMULATIVE_PROMPTS=0

# Check if there are previous checkpoints to resume from
for r in $(seq $NUM_ROUNDS -1 1); do
    LAST_EXP_NAME=qwen15-online-grpo-kl-${KL_COEF}-lr-${LR}-clip-${CLIP_RANGE}-deepscaler-R${r}
    LAST_CKPT=${CACHE_DIR}/${LAST_EXP_NAME}/FINAL
    if [ -d "$LAST_CKPT" ]; then
        CURRENT_CKPT=$LAST_CKPT
        ROUND=$((r + 1))
        CUMULATIVE_PROMPTS=$((r * PROMPTS_PER_ROUND))
        echo "Resuming from round $r checkpoint: $CURRENT_CKPT"
        echo "Starting at round $ROUND with $CUMULATIVE_PROMPTS prompts already processed"
        break
    fi
done

while [ $ROUND -le ${NUM_ROUNDS} ]; do
    echo "Starting round $ROUND of ${NUM_ROUNDS} "
    SAMPLES_FILE=${CACHE_DIR}/R${ROUND}_samples.json

    echo "Sampling from current model..."
    # Sample from current model (first round uses llama-3-8b-instruct checkpoint)
    # e.g. python -m train.sample Qwen/Qwen2.5-0.5B-Instruct --output_file /data/niklas/HALOs/models/qwen-online-grpo-kl-0.05-lr-1e-6-clip-0.2-deepscaler/R1_samples.json --gpu_count 2 --datasets deepscaler --mode train --split train --num_samples_per_prompt 8 --num_prompts 256 --num_skip 0 --batch_size 256
    # no top_p as it often leads to infinite loops
    python -m train.sample $CURRENT_CKPT \
        --output_file $SAMPLES_FILE \
        --gpu_count 4 \
        --datasets deepscaler \
        --mode train \
        --split train \
        --num_samples_per_prompt 8 \
        --top_p 1.0 \
        --max_tokens 4096 \
        --num_prompts ${PROMPTS_PER_ROUND} \
        --num_skip $CUMULATIVE_PROMPTS \
        --batch_size ${PROMPTS_PER_ROUND}

    echo "Sampling complete. Labeling samples..."

    # if no more prompts left, end training early
    NUM_SAMPLES=$(jq '. | length' $SAMPLES_FILE)

    if [[ $NUM_SAMPLES -gt 0 ]]; then
        EXP_NAME=qwen15-online-grpo-kl-${KL_COEF}-lr-${LR}-clip-${CLIP_RANGE}-deepscaler-binary-R${ROUND}
        CUMULATIVE_PROMPTS=$((CUMULATIVE_PROMPTS + ${PROMPTS_PER_ROUND}))
        DATA_FILE=${CACHE_DIR}/R${ROUND}_data.json

        # e.g. python -m train.label /data/niklas/HALOs/models/qwen-online-grpo-kl-0.05-lr-1e-6-clip-0.2-deepscaler/R1_samples.json /data/niklas/HALOs/models/qwen-online-grpo-kl-0.05-lr-1e-6-clip-0.2-deepscaler/R1_data.json --verify_type math --feedback_type binary --batch_size 16
        python -m train.label $SAMPLES_FILE $DATA_FILE --verify_type math --feedback_type binary --batch_size 16

        # First round: load from checkpoint
        # Subsequent rounds: policy resumes from previous checkpoint; reference model stays at SFT checkpoint
        if [ $ROUND -eq 1 ]; then
            MODEL_LOAD_ARG="++model.load_from=$CURRENT_CKPT"
        else
            echo "Loading from checkpoint: $CURRENT_CKPT"
            MODEL_LOAD_ARG="++model.from_checkpoint=$CURRENT_CKPT ++model.load_from=$MODEL_PATH"
        fi
        
        # Train PPO model on the newly sampled and labeled data
        # Note that batch size includes the group size
        # 
        accelerate launch --config_file accelerate_config/fsdp_8gpu.yaml --num_processes $NUM_GPUS launch.py loss=grpo model=qwen train_datasets=[$DATA_FILE] test_datasets=[$DATA_FILE] exp_name=${EXP_NAME} \
            ++cache_dir=${CACHE_DIR} \
            ++model.name_or_path=$MODEL_PATH \
            ++lr=${LR} \
            ++intermediate_checkpoints=true \
            ++online=true \
            ++weight_decay=0.0 \
            ++beta1=0.9 \
            ++beta2=0.95 \
            ++eps=1e-5 \
            ++warmup=0.10 \
            ++loss.ppo_epochs=1 \
            ++loss.cliprange=${CLIP_RANGE} \
            ++loss.lam=0.95 \
            ++loss.gamma=1.0 \
            ++loss.critic_coef=0.1 \
            ++loss.KL_coef=${KL_COEF} \
            ++n_epochs=1 \
            ++model.batch_size=128 \
            ++model.max_grad_norm=1 \
            ++model.max_length=4096 \
            ++model.max_prompt_length=1024 \
            ++model.gradient_accumulation_steps=2 \
            ++model.eval_batch_size=128 \
            ++eval_every=999999 ++do_first_eval=false ++frac_unique_undesirable=0.0 \
            ++humanline=true \
            $MODEL_LOAD_ARG 2>&1 | tee $HALO_DIR/logs/$EXP_NAME.log

        NEW_CKPT=${CACHE_DIR}/${EXP_NAME}/FINAL

        # Clean up old checkpoint directory if it's not the SFT checkpoint
        if [ $CURRENT_CKPT != $MODEL_PATH ] && [ $SLURM_PROCID -eq 0 ]; then
            OLD_EXP_DIR=$(dirname $CURRENT_CKPT)
            echo "Cleaning up $OLD_EXP_DIR"
            rm -rf $OLD_EXP_DIR
        fi

        CURRENT_CKPT=$NEW_CKPT
        ROUND=$((ROUND + 1))
    else
        echo "Ending training early due to zero new samples ..."
        break
    fi
done

set +x