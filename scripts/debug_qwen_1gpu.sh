source /env/bin/start-ctx-user
conda activate gritkto

GPUS_PER_NODE=1
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=1
NODE_RANK=0 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
CKPT=/data/niklas/models/qwen-05b-sft

# Run the training script using srun
accelerate launch \
--machine_rank 0 \
--main_process_ip "$MASTER_ADDR" \
--main_process_port $MASTER_PORT \
--num_processes $WORLD_SIZE \
--role $SLURMD_NODENAME: \
--rdzv_conf rdzv_backend=c10d \
--max_restarts 0 \
launch.py loss=kto model=qwen train_datasets=[mathstepdpo] test_datasets=[mathstepdpo] exp_name=qwen-05B-sft \
++cache_dir=/data/niklas/models \
++model.name_or_path=$MODEL_PATH \
++model.use_peft=false ++cache_reference_logprobs=false ++lr=1e-5 weight_decay=0.0001 ++beta1=0.9 ++beta2=0.95 \
++model.batch_size=2 ++model.gradient_accumulation_steps=1 ++model.microbatch_size=2 ++model.eval_batch_size=2 \
++model.max_length=5000 ++model.max_prompt_length=5000 \
++model.attn_implementation=flash_attention_2 ++eval_every=999999
